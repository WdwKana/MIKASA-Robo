"""
EBM-SRB-MS: Self-Referential Memory + Motion Suppression.

Fixes the failure mode diagnosed in analysis/pem/diag_buffer_dynamics.py:
plain SRB confused "moves every frame" with "novel," so the gripper/arm
dominated buffer writes and the small task-relevant cube was under-
represented. We add a per-patch EMA of frame-to-frame change and divide
the SRB surprise by it:

  s_raw[n]      = min over m in buffer of ‖p_t[n] − buffer[m]‖²
  ema_change[n] ← (1−α) · ema_change[n] + α · ‖p_t[n] − p_{t-1}[n]‖²
  surprise[n]   = s_raw[n] / (1 + λ · ema_change[n] / d_vit)

Intuition: patches that move every frame are motion artifacts (the gripper);
patches that reveal once and then stay stable are task-relevant events (the
cube after the occluder lifts). MS keeps the second and crushes the first.

Stage-0 cube-preservation on cached RC9 frames:
  V1 saliency head      early 0.90  late 0.35
  SRB original          early 0.52  late 0.36
  SRB-MS (this module)  early 0.80  late 0.93   ← best of all three

`p_prev` and `ema_change` are registered buffers reset at episode
boundaries alongside `gru_state`. API matches the SRB v1 module so the
PPO entry is a 1-line import swap.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .frozen_vit import FrozenDualDinoV2
from .frozen_clip import FrozenDualClip
from .frozen_clip_v2 import FrozenDualClipV2
from .episodic_buffer import EpisodicBuffer
from .memory_reader import MemoryReader


def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Efficient (B, N, M) squared L2 distance between a (B,N,d) and b (B,M,d).
    Uses ||x-y||² = ||x||² + ||y||² − 2 x·y."""
    a_norm = (a * a).sum(-1, keepdim=True)           # (B, N, 1)
    b_norm = (b * b).sum(-1).unsqueeze(1)            # (B, 1, M)
    inner = torch.bmm(a, b.transpose(1, 2))          # (B, N, M)
    return (a_norm + b_norm - 2.0 * inner).clamp_min(0.0)


class EBMSRBMSMemoryModule(nn.Module):
    """Self-Referential Buffer + Motion Suppression + LSTM working memory +
    V1-style buffer/reader/fuse.

    `saliency_ckpt`, `no_saliency` accepted for API parity but IGNORED.

    New args vs SRB v1:
      ms_alpha   : EMA decay for per-patch frame-to-frame change (default 0.1)
      ms_lambda  : suppression strength (default 1.0); higher → arm crushed more
    """

    def __init__(
        self,
        num_envs: int,
        proprio_dim: int = 25,
        saliency_ckpt: str | Path | None = None,    # unused (API parity)
        vit_backbone: str = "dinov2",
        L: int = 64,
        K: int = 8,
        d_proj: int = 256,
        d_state: int = 256,
        novelty_thresh: float = 0.95,
        tau_age: float = 30.0,
        gru_hidden_size: int = 128,
        device: str | torch.device = "cuda",
        no_saliency: bool = False,                  # unused (API parity)
        ms_alpha: float = 0.1,
        ms_lambda: float = 1.0,
    ):
        super().__init__()
        self.num_envs = num_envs
        self.K = K
        self.L = L
        self.d_state = d_state
        self.device = torch.device(device)
        self.vit_backbone = vit_backbone
        self.gru_hidden_size = gru_hidden_size
        self.ms_alpha = float(ms_alpha)
        self.ms_lambda = float(ms_lambda)
        H = gru_hidden_size

        # ── frozen perception ─────────────────────────────────────────────
        if vit_backbone == "dinov2":
            self.vit = FrozenDualDinoV2()
        elif vit_backbone == "clip":
            self.vit = FrozenDualClip()
        elif vit_backbone == "clip_v2":
            self.vit = FrozenDualClipV2()
        else:
            raise ValueError(f"unknown vit_backbone: {vit_backbone}")
        self.vit.to(self.device)
        d_vit = self.vit.dim

        # NO saliency head, NO xy_concat — that's the whole point.

        # ── learned: current-frame summary ────────────────────────────────
        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        summary_dim = 128

        # ── buffer (per-env, non-trainable state) ─────────────────────────
        self.buffer = EpisodicBuffer(
            num_envs=num_envs, L=L, d=d_vit, device=self.device,
            tau_age=tau_age, novelty_thresh=novelty_thresh,
        )

        # ── memory reader ─────────────────────────────────────────────────
        self.reader = MemoryReader(
            d_query=proprio_dim + summary_dim, d_buffer=d_vit, d_proj=d_proj,
        )

        # ── LSTM working memory (same shape trick as ebm_hybrid_lstm) ─────
        # gru_input = [pooled top-K patch features, curr, proprio]
        self.gru_input_dim = d_vit + summary_dim + proprio_dim
        self.lstm = nn.LSTM(self.gru_input_dim, H, num_layers=1, batch_first=False)
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)
        # flat state: [h ; c] concatenated, shape (1, B, 2H)
        self.register_buffer("gru_state",
                             torch.zeros(1, num_envs, 2 * H, device=self.device))

        # ── motion-suppression state ──────────────────────────────────────
        # N_total = 2 * (input_size/patch_size)^2 (dual-view DINOv2)
        N_total = 2 * (self.vit.input_size // self.vit.patch_size) ** 2
        self.register_buffer("p_prev",
                             torch.zeros(num_envs, N_total, d_vit, device=self.device))
        self.register_buffer("ema_change",
                             torch.zeros(num_envs, N_total, device=self.device))
        self.register_buffer("ms_initialized",
                             torch.zeros(num_envs, dtype=torch.bool, device=self.device))
        self._d_vit_norm = float(d_vit)

        # ── fuse ──────────────────────────────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj + H, 256),
            nn.GELU(),
            nn.Linear(256, d_state),
        )

    # ─── LSTM state plumbing (h, c packed into one buffer) ─────────────────

    def _split_state(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.gru_hidden_size
        return flat[..., :H].contiguous(), flat[..., H:].contiguous()

    def _join_state(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, c], dim=-1)

    # ─── per-step forward ──────────────────────────────────────────────────

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        # 1. ViT (no_grad)
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)    # (B, N=162, d_vit)
        B, N, d = tokens_all.shape

        # 2. Self-referential surprise + motion suppression.
        with torch.no_grad():
            buf = self.buffer.features                   # (B, L, d), pre-push
            mask = self.buffer.mask                      # (B, L) bool, true=valid
            any_used = mask.any(dim=-1)                  # (B,)

            # pairwise dist (B, N, L)
            dist = _pairwise_sq_dist(tokens_all, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values           # (B, N)
            fallback = (tokens_all * tokens_all).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            # Update per-patch EMA of frame-to-frame change.
            # `ms_initialized[b]` is False at episode start; we skip the
            # change update on that step and just record p_t as p_prev.
            init_now = self.ms_initialized                            # (B,)
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)    # (B, N)
            # only update ema where this env's previous frame exists
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            # store p_prev for next step and mark env as initialized
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # apply motion suppression
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            surprise = s_raw / suppress

            # top-K candidates
            topk_val, topk_idx = surprise.topk(self.K, dim=-1)
            cand_feats = tokens_all.gather(
                1, topk_idx.unsqueeze(-1).expand(B, self.K, d))
            cand_priority = topk_val                     # high surprise → high priority

        # 3. Push to buffer (uses cand_priority for novelty filter + eviction)
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

        # 4. curr_summary + pooled top-K (gru context)
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        # softmax-weighted pool over K (priority-weighted) — same trick as V3a
        pool_w = torch.softmax(cand_priority, dim=-1).unsqueeze(-1)   # (B, K, 1)
        pooled = (pool_w * cand_feats).sum(dim=1)                     # (B, d_vit)

        # 5. LSTM step (no_grad here; replay handles gradients)
        gru_input = torch.cat([pooled, curr, proprio], dim=-1).unsqueeze(0)  # (1, B, gin)
        with torch.no_grad():
            h, c = self._split_state(self.gru_state)
            _, (new_h, new_c) = self.lstm(gru_input, (h, c))
            self.gru_state = self._join_state(new_h, new_c).detach()
        gru_hidden = new_h.squeeze(0)                                  # (B, H)

        # 6. Cross-attn read (post-push buffer state)
        feats, mask_r, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask_r, ts, sal)      # (B, d_proj)

        # 7. Fuse
        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "max_surprise": topk_val.max(-1).values,
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),
            "gru_state_post": self.gru_state.detach().clone(),
        }

    # ─── replay (PPO update; gradients flow through reader + LSTM + fuse) ──

    def replay(self, cached_buffer: dict,
               cls_base: torch.Tensor, cls_hand: torch.Tensor,
               proprio: torch.Tensor,
               gru_state_pre: torch.Tensor,
               gru_input: torch.Tensor) -> torch.Tensor:
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

        h, c = self._split_state(gru_state_pre)
        _, (new_h, _) = self.lstm(gru_input.unsqueeze(0), (h, c))
        gru_hidden = new_h.squeeze(0)

        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))
        return s_t

    # ─── episode boundary + snapshot/restore (PPO infra) ───────────────────

    def reset(self, env_done_mask: torch.Tensor) -> None:
        self.buffer.reset(env_done_mask)
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if env_done_mask.any():
            self.gru_state[:, env_done_mask] = 0.0
            # also reset motion-suppression state so the next step's surprise
            # is computed against this episode's history, not the previous one
            self.p_prev[env_done_mask] = 0.0
            self.ema_change[env_done_mask] = 0.0
            self.ms_initialized[env_done_mask] = False

    def snapshot(self) -> dict:
        return {
            "buffer":      self.buffer.state_dict_buffer(),
            "gru_state":   self.gru_state.clone(),
            "p_prev":      self.p_prev.clone(),
            "ema_change":  self.ema_change.clone(),
            "ms_init":     self.ms_initialized.clone(),
        }

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd["buffer"])
        self.gru_state.copy_(sd["gru_state"])
        if "p_prev" in sd:                       # backwards compat
            self.p_prev.copy_(sd["p_prev"])
            self.ema_change.copy_(sd["ema_change"])
            self.ms_initialized.copy_(sd["ms_init"])
