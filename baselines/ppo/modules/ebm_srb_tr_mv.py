"""
EBM-SRB-TR-MV: SRB-TR with Multi-View Split-K buffer writes.

Adds a parallel RGB-space surprise channel to SRB-TR's buffer-write routing.
Motivated by an empirical finding: per-patch features of all large frozen ViT
backbones we have access to fail to separate MIKASA's saturated synthetic
colors under L2 distance:

    DINOv2-small : sep_ratio = 1.00
    CLIP-base    : sep_ratio = 1.07
    SAM-base     : sep_ratio = 1.04

This blindness causes SRB-MS / SRB-TR to coalesce different-color cubes into
the same buffer slot on RememberColor* tasks. We restore color
discriminability without retraining the backbone or replacing it: each step,
buffer-write candidates are picked in a split fashion — K/2 by the DINOv2
SRB-MS score (semantic novelty) and K/2 by an RGB-mean SRB-MS score (color
novelty), then de-duplicated before push.

RGB reference is a FIFO of size L (mirrors buffer size, but not slot-aligned
with episodic buffer because EpisodicBuffer's priority-eviction policy is
distinct from RGB FIFO's recency policy). Motion suppression is tracked
separately for the RGB channel.

LSTM routing (high-motion patches → working memory) is unchanged from SRB-TR.

API parity with SRB-TR; PPO entry is a 1-line import swap.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ebm_srb_tr import EBMSRBTRMemoryModule, _pairwise_sq_dist


class EBMSRBTRMVMemoryModule(EBMSRBTRMemoryModule):
    """SRB-TR + RGB-view split-K buffer writes."""

    def __init__(
        self,
        *args,
        mv_k_dinov2: int | None = None,   # default: K // 2
        mv_k_rgb: int | None = None,      # default: K // 2
        rgb_hist_size: int | None = None, # default: same as buffer L
        ms_rgb_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Split-K config
        K = self.K
        self.mv_k_dinov2 = mv_k_dinov2 if mv_k_dinov2 is not None else K // 2
        self.mv_k_rgb    = mv_k_rgb    if mv_k_rgb    is not None else K - self.mv_k_dinov2
        assert self.mv_k_dinov2 + self.mv_k_rgb >= 1
        self.ms_rgb_lambda = float(ms_rgb_lambda)

        # RGB FIFO reference (per-env)
        B = self.num_envs
        M = rgb_hist_size if rgb_hist_size is not None else self.L
        self.rgb_hist_size = int(M)
        self.register_buffer("rgb_hist",
                             torch.zeros(B, self.rgb_hist_size, 3, device=self.device))
        self.register_buffer("rgb_hist_used",
                             torch.zeros(B, dtype=torch.long, device=self.device))
        self.register_buffer("rgb_hist_ptr",
                             torch.zeros(B, dtype=torch.long, device=self.device))

        # Per-patch RGB motion-suppression state (mirrors DINOv2 side)
        N_total = 2 * (self.vit.input_size // self.vit.patch_size) ** 2
        self.register_buffer("rgb_prev",
                             torch.zeros(B, N_total, 3, device=self.device))
        self.register_buffer("ema_rgb_change",
                             torch.zeros(B, N_total, device=self.device))
        # rgb_initialized shares lifecycle with ms_initialized

    # ─── helpers ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _rgb_per_patch(self, rgb6: torch.Tensor) -> torch.Tensor:
        """Compute per-patch mean RGB matching the DINOv2 grid.

        rgb6: (B, H, W, 6) uint8 or float in [0,255]
        Returns (B, N_total, 3) float in [0,1].
        """
        x = rgb6
        if x.dtype == torch.uint8:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()              # (B, 6, H, W)
        base = x[:, :3]
        hand = x[:, 3:]
        S = self.vit.input_size                              # 126
        P = self.vit.patch_size                              # 14
        Hp = S // P                                          # 9
        if base.shape[-1] != S:
            base = F.interpolate(base, size=(S, S), mode="bilinear", align_corners=False)
            hand = F.interpolate(hand, size=(S, S), mode="bilinear", align_corners=False)
        B = base.shape[0]
        # (B, 3, Hp, P, Hp, P) → mean over P dims → (B, 3, Hp, Hp)
        base = base.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        hand = hand.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        base = base.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        hand = hand.permute(0, 2, 3, 1).reshape(B, Hp * Hp, 3)
        return torch.cat([base, hand], dim=1)                # (B, 2*Hp*Hp, 3)

    @torch.no_grad()
    def _rgb_surprise(self, rgb_patches: torch.Tensor) -> torch.Tensor:
        """L2 surprise of each patch vs current rgb_hist FIFO contents.

        rgb_patches: (B, N, 3)
        Returns (B, N) — min L2² to FIFO; for empty FIFO returns ||rgb||² fallback.
        """
        B, N, _ = rgb_patches.shape
        idx = torch.arange(self.rgb_hist_size, device=self.device).unsqueeze(0)
        mask = idx < self.rgb_hist_used.unsqueeze(1)         # (B, M) bool
        any_used = mask.any(dim=-1)                          # (B,)
        # pairwise (B, N, M)
        dist = _pairwise_sq_dist(rgb_patches, self.rgb_hist)
        dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
        min_dist = dist.min(dim=-1).values                   # (B, N)
        fallback = (rgb_patches * rgb_patches).sum(-1)
        return torch.where(any_used.unsqueeze(-1), min_dist, fallback)

    @torch.no_grad()
    def _push_rgb_fifo(self, rgb_patches: torch.Tensor,
                       idx: torch.Tensor) -> None:
        """Push the K_rgb selected RGB patches into FIFO. idx: (B, K_rgb) long."""
        B, K = idx.shape
        if K == 0: return
        # Gather: (B, K, 3)
        selected = rgb_patches.gather(1, idx.unsqueeze(-1).expand(B, K, 3))
        M = self.rgb_hist_size
        for k in range(K):
            slot = self.rgb_hist_ptr % M                     # (B,)
            self.rgb_hist[torch.arange(B, device=self.device), slot] = selected[:, k]
            self.rgb_hist_ptr = (self.rgb_hist_ptr + 1) % M
            self.rgb_hist_used = torch.clamp(self.rgb_hist_used + 1, max=M)

    # ─── per-step forward (override) ──────────────────────────────────────

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int):
        # 1. ViT
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)        # (B, N, d_vit)
        B, N, d = tokens_all.shape

        # 2. Per-patch RGB
        rgb_patches = self._rgb_per_patch(rgb6)              # (B, N, 3)

        with torch.no_grad():
            # ── DINOv2 SRB-MS surprise (unchanged) ────────────────────────
            buf = self.buffer.features
            mask = self.buffer.mask
            any_used = mask.any(dim=-1)
            dist = _pairwise_sq_dist(tokens_all, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values
            fallback = (tokens_all * tokens_all).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            # motion suppression: DINOv2 side
            init_now = self.ms_initialized
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            # motion suppression: RGB side
            cur_rgb_change = ((rgb_patches - self.rgb_prev) ** 2).sum(-1)
            upd_r = self.ms_alpha * cur_rgb_change + (1 - self.ms_alpha) * self.ema_rgb_change
            self.ema_rgb_change = torch.where(init_now.unsqueeze(-1), upd_r, self.ema_rgb_change)

            self.p_prev = tokens_all.detach()
            self.rgb_prev = rgb_patches.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # ── Buffer write routing (multi-view split-K) ─────────────────
            # Branch A (DINOv2): low-motion semantic-novel patches
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            dinov2_score = s_raw / suppress                  # (B, N)
            # Branch B (RGB): low-motion color-novel patches
            rgb_raw = self._rgb_surprise(rgb_patches)        # (B, N)
            rgb_suppress = 1.0 + self.ms_rgb_lambda * self.ema_rgb_change / 3.0
            rgb_score = rgb_raw / rgb_suppress               # (B, N)

            # Top-K_dinov2 and Top-K_rgb independently
            kd = self.mv_k_dinov2
            kr = self.mv_k_rgb
            v_d, i_d = dinov2_score.topk(kd, dim=-1)         # (B, kd)
            v_r, i_r = rgb_score.topk(kr, dim=-1)            # (B, kr)

            # Dedup union per env (pad to K with the highest leftover dinov2 score)
            K = self.K
            cand_idx = torch.zeros(B, K, dtype=torch.long, device=self.device)
            cand_prio = torch.zeros(B, K, device=self.device)
            for b in range(B):
                merged = []
                seen = set()
                # interleave dinov2 / rgb picks to balance
                for k in range(max(kd, kr)):
                    if k < kd:
                        n = int(i_d[b, k].item())
                        if n not in seen:
                            seen.add(n); merged.append((n, float(dinov2_score[b, n].item())))
                    if k < kr:
                        n = int(i_r[b, k].item())
                        if n not in seen:
                            seen.add(n); merged.append((n, float(dinov2_score[b, n].item())))
                # pad with next-best dinov2 picks if dedup ate slots
                if len(merged) < K:
                    extra_v, extra_i = dinov2_score[b].topk(K + kd, dim=-1)
                    for j in range(K + kd):
                        n = int(extra_i[j].item())
                        if n in seen: continue
                        seen.add(n); merged.append((n, float(extra_v[j].item())))
                        if len(merged) >= K: break
                merged = merged[:K]
                for k, (n, p) in enumerate(merged):
                    cand_idx[b, k] = n
                    cand_prio[b, k] = p

            cand_feats = tokens_all.gather(1, cand_idx.unsqueeze(-1).expand(B, K, d))
            cand_rgb = rgb_patches.gather(1, cand_idx.unsqueeze(-1).expand(B, K, 3))

            # ── LSTM routing (unchanged): high-motion patches → working mem ─
            topk_val_lstm, topk_idx_lstm = self.ema_change.topk(self.K, dim=-1)
            lstm_feats = tokens_all.gather(
                1, topk_idx_lstm.unsqueeze(-1).expand(B, self.K, d))

        # 3. Push to episodic buffer + update RGB FIFO
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_prio, t_now=t)
            # FIFO update: push only the K_rgb slots from rgb-side picks (so the
            # RGB reference reflects what RGB-novelty surfaced; DINOv2-only picks
            # may or may not be color-novel, so we don't want them to crowd FIFO).
            # Use rgb-side originals (pre-dedup) so we always push kr items.
            self._push_rgb_fifo(rgb_patches, i_r)
        n_pushed = (cand_prio > 0).sum(dim=-1)

        # 4-7. Rest mirrors SRB-TR
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
            "max_surprise": cand_prio.max(-1).values,
            "max_motion":   topk_val_lstm.max(-1).values,
            "rgb_hist_used": self.rgb_hist_used.clone(),
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),
            "gru_state_post": self.gru_state.detach().clone(),
        }

    # ─── episode boundary + snapshot/restore ──────────────────────────────

    def reset(self, env_done_mask: torch.Tensor) -> None:
        super().reset(env_done_mask)
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if env_done_mask.any():
            self.rgb_hist[env_done_mask] = 0.0
            self.rgb_hist_used[env_done_mask] = 0
            self.rgb_hist_ptr[env_done_mask] = 0
            self.rgb_prev[env_done_mask] = 0.0
            self.ema_rgb_change[env_done_mask] = 0.0

    def snapshot(self) -> dict:
        sd = super().snapshot()
        sd["rgb_hist"]      = self.rgb_hist.clone()
        sd["rgb_hist_used"] = self.rgb_hist_used.clone()
        sd["rgb_hist_ptr"]  = self.rgb_hist_ptr.clone()
        sd["rgb_prev"]      = self.rgb_prev.clone()
        sd["ema_rgb_change"] = self.ema_rgb_change.clone()
        return sd

    def restore(self, sd: dict) -> None:
        super().restore(sd)
        if "rgb_hist" in sd:
            self.rgb_hist.copy_(sd["rgb_hist"])
            self.rgb_hist_used.copy_(sd["rgb_hist_used"])
            self.rgb_hist_ptr.copy_(sd["rgb_hist_ptr"])
            self.rgb_prev.copy_(sd["rgb_prev"])
            self.ema_rgb_change.copy_(sd["ema_rgb_change"])
