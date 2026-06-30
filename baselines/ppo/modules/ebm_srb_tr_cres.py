"""
EBM-SRB-TR-CRES: Color-RESidual buffer (GPT fusion form 1).

A color-aware variant of SRB-TR motivated by the same finding as SRB-TR-MV:
frozen DINOv2 per-patch features do not separate MIKASA's saturated synthetic
colors under L2 / cosine, so the episodic buffer coalesces different-color
cubes into the same slot on RememberColor* tasks.

Where SRB-TR-MV injects color at the SCORING layer (a parallel RGB-novelty
channel decides *which* patches to write, but the stored features stay pure
DINOv2), CRES injects color at the FEATURE layer, the way GPT's "color-aware
adapter" proposes — additive residual:

    z = z_dino + s · (meanRGB_patch @ R)

with R a FIXED random projection (3 → d_vit) and s a single scale frozen after
the first step so that the color residual has norm ≈ `color_frac` × ‖z_dino‖.
The fused z (still d_vit-dimensional) is what the buffer scores (L2 surprise),
filters (cosine novelty), stores, and what the reader later attends over. So
same-color patches stay close and different-color patches separate IN THE
STORED REPRESENTATION — not just in the write decision.

Dimensionality is preserved (d_vit), so the PPO replay cache (which is sized to
viT dim) is unchanged: the PPO entry is a pure 1-line import swap.

Design choices that keep this surgical:
  • Motion suppression (`ema_change`) and the LSTM working-memory route stay on
    PURE DINOv2 features — color is static across frames and should not perturb
    the motion timescale split.
  • `curr_summary` uses the CLS tokens unchanged.
  • R uses a fixed generator seed so the train-side and eval-side module
    instances build the IDENTICAL color subspace (the shared reader is trained
    against one R; a mismatched eval R would be read as noise).

Known risk (documented in analysis/color_head/COLOR_PERCEPTION_FINDINGS.md §4):
feature-level injection can over-tighten same-color clusters and trip the
novelty filter, collapsing buffer diversity on RC9. CRES keeps `color_frac`
modest and additive (not contrastive) to stay in the regime where DINOv2
geometry still dominates; RC9 must be re-checked, but the target task here is
RC5, where the failure mode is the opposite (colors not separable enough).
"""
from __future__ import annotations

import torch

from .ebm_srb_tr import EBMSRBTRMemoryModule, _pairwise_sq_dist


class EBMSRBTRCRESMemoryModule(EBMSRBTRMemoryModule):
    """SRB-TR + additive per-patch color residual on the stored features."""

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
        *increases* their cosine (verified counterproductive in
        analysis/color_head/probe_dino_norm.py). Centering puts colors on
        signed axes so distinct hues get near-orthogonal residuals, and neutral
        (~gray table, ≈0.5) patches get ≈0 residual — color perturbs only the
        saturated cube patches, leaving the table geometry intact.
        """
        rgb_patches = self._rgb_per_patch(rgb6) - 0.5         # (B, N, 3) centered
        rgb_proj = torch.einsum("bnc,cd->bnd", rgb_patches, self.color_R)  # (B, N, d)
        if self.color_scale.item() < 0:
            mean_dino = tokens_all.norm(dim=-1).mean()
            mean_proj = rgb_proj.norm(dim=-1).mean().clamp_min(1e-6)
            self.color_scale.fill_(float(self.color_frac * mean_dino / mean_proj))
        return tokens_all + self.color_scale * rgb_proj

    # ─── per-step forward (override) ───────────────────────────────────────
    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int):
        # 1. ViT
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)         # (B, N, d_vit)
        B, N, d = tokens_all.shape

        with torch.no_grad():
            # 2. Color-fused feature used for ALL buffer operations.
            z = self._fuse_color(tokens_all, rgb6)            # (B, N, d_vit)

            # 3. Self-referential surprise on z vs buffer (which holds z's).
            buf = self.buffer.features
            mask = self.buffer.mask
            any_used = mask.any(dim=-1)
            dist = _pairwise_sq_dist(z, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values
            fallback = (z * z).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            # 4. Motion suppression on PURE DINOv2 features (color is static).
            init_now = self.ms_initialized
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # 5. Buffer route: low-motion novel patches (SRB-MS), stored as z.
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            buffer_score = s_raw / suppress
            topk_val_buf, topk_idx_buf = buffer_score.topk(self.K, dim=-1)
            cand_feats = z.gather(1, topk_idx_buf.unsqueeze(-1).expand(B, self.K, d))
            cand_priority = topk_val_buf

            # 6. LSTM route: high-motion patches, pooled from PURE features.
            topk_val_lstm, topk_idx_lstm = self.ema_change.topk(self.K, dim=-1)
            lstm_feats = tokens_all.gather(
                1, topk_idx_lstm.unsqueeze(-1).expand(B, self.K, d))

        # 7. Push color-aware features to buffer.
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

        # 8-10. Rest mirrors SRB-TR exactly.
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        pool_w = torch.softmax(topk_val_lstm, dim=-1).unsqueeze(-1)
        pooled = (pool_w * lstm_feats).sum(dim=1)

        gru_input = torch.cat([pooled, curr, proprio], dim=-1).unsqueeze(0)
        with torch.no_grad():
            h, c = self._split_state(self.gru_state)
            _, (new_h, new_c) = self.lstm(gru_input, (h, c))
            self.gru_state = self._join_state(new_h, new_c).detach()
        gru_hidden = new_h.squeeze(0)

        feats, mask_r, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask_r, ts, sal)

        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "max_surprise": topk_val_buf.max(-1).values,
            "max_motion":   topk_val_lstm.max(-1).values,
            "color_scale": self.color_scale.clone(),
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),
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
