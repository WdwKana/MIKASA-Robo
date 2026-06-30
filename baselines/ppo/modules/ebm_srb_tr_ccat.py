"""
EBM-SRB-TR-CCAT: Color-CONCAT buffer (GPT fusion form 2).

Sibling of SRB-TR-CRES. Same motivation and same injection LOCUS (the episodic
buffer's stored representation), but GPT's *other* fusion form — concatenation
instead of additive residual:

    z = [ z_dino (d_vit) ,  s · (meanRGB_patch − 0.5)  (3) ]      ∈ ℝ^(d_vit+3)

The buffer scores (L2 surprise), filters (cosine novelty), stores, and the
reader attends over this (d_vit+3)-dim vector. Versus CRES, the color signal
lives in dedicated, un-entangled dimensions, so the learned reader (W_K/W_V)
can read color off cleanly rather than disentangling it from a random
projection — at the cost of changing the buffer dimensionality.

`s` is a single scale frozen after the first step so the color block carries
norm ≈ `color_frac` × ‖z_dino‖ (calibration verified in
analysis/color_head/probe_dino_norm.py; centered RGB at color_frac≈0.4 widens
the same-vs-different-color cosine gap and pushes different-color cube patches
below the 0.95 novelty threshold without over-tightening same-color clusters).

Because the buffer dimension is now d_vit+3, the PPO entry must size its replay
cache to `agent.ebm.buffer.d` (2 lines vs the CRES import-only swap); see
baselines/ppo/ppo_memtasks_ebm_srb_tr_ccat.py.

Motion suppression and the LSTM working-memory route stay on PURE DINOv2
features (color is static); curr_summary uses CLS tokens unchanged.
"""
from __future__ import annotations

import torch

from .ebm_srb_tr import EBMSRBTRMemoryModule, _pairwise_sq_dist
from .episodic_buffer import EpisodicBuffer
from .memory_reader import MemoryReader


class EBMSRBTRCCATMemoryModule(EBMSRBTRMemoryModule):
    """SRB-TR with per-patch centered-RGB concatenated into stored features."""

    N_COLOR = 3

    def __init__(self, *args, color_frac: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_frac = float(color_frac)
        d_vit = self.vit.dim
        d_buf = d_vit + self.N_COLOR

        # Rebuild buffer + reader at the widened dimension. (Parent built them
        # at d_vit; the learned reader is re-initialized fresh, which is fine —
        # training starts from scratch anyway.)
        tau_age = self.buffer.tau_age
        novelty_thresh = self.buffer.novelty_thresh
        self.buffer = EpisodicBuffer(
            num_envs=self.num_envs, L=self.L, d=d_buf, device=self.device,
            tau_age=tau_age, novelty_thresh=novelty_thresh,
        )
        d_query = self.reader.W_Q.in_features
        d_proj = self.reader.W_Q.out_features
        alpha = self.reader.alpha
        self.reader = MemoryReader(d_query=d_query, d_buffer=d_buf,
                                   d_proj=d_proj, alpha=alpha).to(self.device)

        self.register_buffer("color_scale", torch.tensor(-1.0, device=self.device))

    # ─── per-patch mean RGB (matches DINOv2 9x9 grid, both views) ──────────
    @torch.no_grad()
    def _rgb_per_patch(self, rgb6: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        x = rgb6
        if x.dtype == torch.uint8:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        base, hand = x[:, :3], x[:, 3:]
        S = self.vit.input_size
        P = self.vit.patch_size
        Hp = S // P
        if base.shape[-1] != S:
            base = F.interpolate(base, size=(S, S), mode="bilinear", align_corners=False)
            hand = F.interpolate(hand, size=(S, S), mode="bilinear", align_corners=False)
        B = base.shape[0]
        base = base.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        hand = hand.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        base = base.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        hand = hand.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        return torch.cat([base, hand], dim=1)                 # (B, N, 3)

    @torch.no_grad()
    def _fuse_color(self, tokens_all: torch.Tensor, rgb6: torch.Tensor) -> torch.Tensor:
        """z = [tokens_all, s·(meanRGB-0.5)]. Calibrate s once, then freeze."""
        c = self._rgb_per_patch(rgb6) - 0.5                   # (B, N, 3) centered
        if self.color_scale.item() < 0:
            mean_dino = tokens_all.norm(dim=-1).mean()
            mean_c = c.norm(dim=-1).mean().clamp_min(1e-6)
            self.color_scale.fill_(float(self.color_frac * mean_dino / mean_c))
        return torch.cat([tokens_all, self.color_scale * c], dim=-1)

    # ─── per-step forward (override) ───────────────────────────────────────
    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int):
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)         # (B, N, d_vit)
        B, N, d_vit = tokens_all.shape

        with torch.no_grad():
            z = self._fuse_color(tokens_all, rgb6)            # (B, N, d_buf)
            d_buf = z.shape[-1]

            # surprise on z vs buffer (holds z's)
            buf = self.buffer.features
            mask = self.buffer.mask
            any_used = mask.any(dim=-1)
            dist = _pairwise_sq_dist(z, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values
            fallback = (z * z).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            # motion suppression on PURE DINOv2 features
            init_now = self.ms_initialized
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # buffer route: low-motion novel patches, stored as z (d_buf)
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            buffer_score = s_raw / suppress
            topk_val_buf, topk_idx_buf = buffer_score.topk(self.K, dim=-1)
            cand_feats = z.gather(1, topk_idx_buf.unsqueeze(-1).expand(B, self.K, d_buf))
            cand_priority = topk_val_buf

            # LSTM route: high-motion patches, pooled from PURE features
            topk_val_lstm, topk_idx_lstm = self.ema_change.topk(self.K, dim=-1)
            lstm_feats = tokens_all.gather(
                1, topk_idx_lstm.unsqueeze(-1).expand(B, self.K, d_vit))

        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

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

    def snapshot(self) -> dict:
        sd = super().snapshot()
        sd["color_scale"] = self.color_scale.clone()
        return sd

    def restore(self, sd: dict) -> None:
        super().restore(sd)
        if "color_scale" in sd:
            self.color_scale.copy_(sd["color_scale"])
