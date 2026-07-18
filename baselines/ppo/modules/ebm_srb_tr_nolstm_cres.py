"""SRB-TR-NoLSTM + CRES: the buffer-only ablation with the color residual.

Composition of two existing modules, for the Remember-family NoLSTM ablation:
  - base:  ebm_srb_tr_nolstm.EBMSRBTRNoLSTMMemoryModule  (LSTM branch removed)
  - CRES:  ebm_srb_tr_cres.EBMSRBTRCRESMemoryModule       (color residual)

`__init__` extras, `_rgb_per_patch`, `_fuse_color`, `snapshot`/`restore` are
copied VERBATIM from ebm_srb_tr_cres.py. `step` is the NoLSTM step with the
same substitutions cres makes to the base step (and nothing else):
  * z = _fuse_color(tokens_all) computed right after the ViT;
  * surprise distance + fallback computed on z (buffer stores z's);
  * candidate features gathered from z;
  * motion suppression stays on PURE DINOv2 tokens (color is static).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.ppo.modules.ebm_srb_tr_nolstm import (
    EBMSRBTRNoLSTMMemoryModule,
    _pairwise_sq_dist,
)


class EBMSRBTRNoLSTMCRESMemoryModule(EBMSRBTRNoLSTMMemoryModule):
    """SRB-TR-NoLSTM + additive per-patch color residual on stored features."""

    def __init__(self, *args, color_frac: float = 0.4, proj_seed: int = 1234,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.color_frac = float(color_frac)
        d_vit = self.vit.dim
        # Fixed random projection 3 -> d_vit. Variance 1/d_vit so that for a
        # unit-norm RGB input ||rgb @ R|| ≈ ||rgb|| (keeps `s` interpretable).
        g = torch.Generator().manual_seed(int(proj_seed))
        R = torch.randn(3, d_vit, generator=g) / (d_vit ** 0.5)
        self.register_buffer("color_R", R.to(self.device))
        # Scale frozen after first step (lazy auto-calibration). -1 = uncalibrated.
        self.register_buffer("color_scale", torch.tensor(-1.0, device=self.device))

    # ─── per-patch mean RGB (matches DINOv2 9x9 grid, both views) ──────────
    @torch.no_grad()
    def _rgb_per_patch(self, rgb6: torch.Tensor) -> torch.Tensor:
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
            import torch.nn.functional as F
            base = F.interpolate(base, size=(S, S), mode="bilinear", align_corners=False)
            hand = F.interpolate(hand, size=(S, S), mode="bilinear", align_corners=False)
        B = base.shape[0]
        base = base.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        hand = hand.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        base = base.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        hand = hand.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        return torch.cat([base, hand], dim=1)                 # (B, N_total, 3)

    @torch.no_grad()
    def _fuse_color(self, tokens_all: torch.Tensor, rgb6: torch.Tensor) -> torch.Tensor:
        """z = tokens_all + s · ((meanRGB-0.5) @ R). Calibrate s once, then freeze.

        Centering by 0.5 is essential: raw RGB ∈ [0,1]³ is all-positive, so
        different colors project into a shared positive cone and the residual
        *increases* their cosine. Centering puts colors on signed axes so
        distinct hues get near-orthogonal residuals, and neutral (~gray table,
        ≈0.5) patches get ≈0 residual.
        """
        rgb_patches = self._rgb_per_patch(rgb6) - 0.5         # (B, N, 3) centered
        rgb_proj = torch.einsum("bnc,cd->bnd", rgb_patches, self.color_R)  # (B, N, d)
        if self.color_scale.item() < 0:
            mean_dino = tokens_all.norm(dim=-1).mean()
            mean_proj = rgb_proj.norm(dim=-1).mean().clamp_min(1e-6)
            self.color_scale.fill_(float(self.color_frac * mean_dino / mean_proj))
        return tokens_all + self.color_scale * rgb_proj

    # ─── per-step forward (override: NoLSTM step + CRES substitutions) ─────
    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        # 1. ViT (no_grad)
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)    # (B, N=162, d_vit)
        B, N, d = tokens_all.shape

        with torch.no_grad():
            # 2. Color-fused feature used for ALL buffer operations.
            z = self._fuse_color(tokens_all, rgb6)            # (B, N, d_vit)

            # 3. Self-referential surprise on z vs buffer (which holds z's).
            buf = self.buffer.features                   # (B, L, d), pre-push
            mask = self.buffer.mask                      # (B, L) bool, true=valid
            any_used = mask.any(dim=-1)                  # (B,)

            dist = _pairwise_sq_dist(z, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values           # (B, N)
            fallback = (z * z).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            # 4. Motion suppression on PURE DINOv2 features (color is static).
            init_now = self.ms_initialized                            # (B,)
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)    # (B, N)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # 5. Branch A only: episodic buffer ← low-motion novel z tokens.
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            buffer_score = s_raw / suppress
            topk_val_buf, topk_idx_buf = buffer_score.topk(self.K, dim=-1)
            cand_feats = z.gather(
                1, topk_idx_buf.unsqueeze(-1).expand(B, self.K, d))
            cand_priority = topk_val_buf

        # 6. Push color-aware features to buffer
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

        # 7. curr_summary (no pooled-motion, no LSTM)
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))

        # 8. Cross-attn read (post-push buffer state)
        feats, mask_r, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask_r, ts, sal)      # (B, d_proj)

        # 9. Fuse (no gru_hidden)
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "max_surprise": topk_val_buf.max(-1).values,
            "max_motion":   self.ema_change.max(-1).values,
            "color_scale": self.color_scale.clone(),
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            # zeroed placeholders (entry caches these; never used in compute)
            "gru_input":   torch.zeros(B, self.gru_input_dim, device=s_t.device),
            "gru_state_post": self.gru_state.detach().clone(),
        }

    # ─── snapshot/restore: include calibration scalar (constant, but safe) ──
    def snapshot(self) -> dict:
        sd = super().snapshot()
        sd["color_scale"] = self.color_scale.clone()
        return sd

    def restore(self, sd: dict) -> None:
        super().restore(sd)
        if "color_scale" in sd:
            self.color_scale.copy_(sd["color_scale"])
