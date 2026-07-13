"""
EBM-SRB-TR-NoLSTM: ablation — the SRB-TR episodic buffer WITHOUT the LSTM
working-memory branch ("buffer-only").

Purpose: answer "is the recurrent working memory necessary?" — most sharply on
the dynamic Intercept tasks, where the moving ball is task-critical and the
buffer alone (which suppresses high-motion writes) should struggle if the
working-memory branch is what carries dynamics.

Deltas vs `ebm_srb_tr.py` (EXACTLY one branch removed, nothing else touched):
  KEPT   : frozen ViT, self-referential surprise, motion EMA, the ENTIRE
           buffer write path (motion-suppressed surprise top-K, cosine dedup,
           recency eviction) — byte-identical scoring; cross-attention reader;
           per-frame CLS summary.
  REMOVED: the high-motion token routing + LSTM working memory. fuse now takes
           [proprio, curr, retrieved] (no gru_hidden). Motion/dynamics reach
           the policy only through the per-frame CLS summary.

API compatibility: the PPO entry caches `gru_state` / `info["gru_input"]` and
calls `replay(..., gru_state_pre, gru_input)`. We keep a dummy all-zeros
`gru_state` buffer, expose `gru_input_dim`, emit zeroed `gru_input` /
`gru_state_post`, and `replay()` ignores the two recurrent args — so the entry
is a 1-line import swap (minus the eval-agent `.lstm` share).
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
    """Efficient (B, N, M) squared L2 distance between a (B,N,d) and b (B,M,d)."""
    a_norm = (a * a).sum(-1, keepdim=True)           # (B, N, 1)
    b_norm = (b * b).sum(-1).unsqueeze(1)            # (B, 1, M)
    inner = torch.bmm(a, b.transpose(1, 2))          # (B, N, M)
    return (a_norm + b_norm - 2.0 * inner).clamp_min(0.0)


class EBMSRBTRNoLSTMMemoryModule(nn.Module):
    """SRB-TR minus the LSTM working-memory branch (buffer-only ablation).

    `saliency_ckpt`, `no_saliency` accepted for API parity but IGNORED.
    `gru_hidden_size` kept only to size the dummy state buffer (API parity).
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

        # ── NO LSTM working memory (the whole point of this ablation) ─────
        # Dummy state + dims kept so the PPO entry's caching plumbing works
        # unchanged; they carry zeros and are never used in computation.
        self.gru_input_dim = d_vit + summary_dim + proprio_dim
        self.register_buffer("gru_state",
                             torch.zeros(1, num_envs, 2 * H, device=self.device))

        # ── motion-suppression state (still needed for BUFFER writes) ─────
        N_total = 2 * (self.vit.input_size // self.vit.patch_size) ** 2
        self.register_buffer("p_prev",
                             torch.zeros(num_envs, N_total, d_vit, device=self.device))
        self.register_buffer("ema_change",
                             torch.zeros(num_envs, N_total, device=self.device))
        self.register_buffer("ms_initialized",
                             torch.zeros(num_envs, dtype=torch.bool, device=self.device))
        self._d_vit_norm = float(d_vit)

        # ── fuse (no gru_hidden input) ────────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj, 256),
            nn.GELU(),
            nn.Linear(256, d_state),
        )

    # ─── per-step forward ──────────────────────────────────────────────────

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        # 1. ViT (no_grad)
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)    # (B, N=162, d_vit)
        B, N, d = tokens_all.shape

        # 2. Self-referential surprise + motion suppression (identical to TR).
        with torch.no_grad():
            buf = self.buffer.features                   # (B, L, d), pre-push
            mask = self.buffer.mask                      # (B, L) bool, true=valid
            any_used = mask.any(dim=-1)                  # (B,)

            dist = _pairwise_sq_dist(tokens_all, buf)
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values           # (B, N)
            fallback = (tokens_all * tokens_all).sum(-1)
            s_raw = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

            init_now = self.ms_initialized                            # (B,)
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)    # (B, N)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # Branch A only: episodic buffer ← low-motion novel tokens.
            suppress = 1.0 + self.ms_lambda * self.ema_change / self._d_vit_norm
            buffer_score = s_raw / suppress
            topk_val_buf, topk_idx_buf = buffer_score.topk(self.K, dim=-1)
            cand_feats = tokens_all.gather(
                1, topk_idx_buf.unsqueeze(-1).expand(B, self.K, d))
            cand_priority = topk_val_buf

        # 3. Push to buffer
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

        # 4. curr_summary (no pooled-motion, no LSTM)
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))

        # 5. Cross-attn read (post-push buffer state)
        feats, mask_r, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask_r, ts, sal)      # (B, d_proj)

        # 6. Fuse (no gru_hidden)
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "max_surprise": topk_val_buf.max(-1).values,
            "max_motion":   self.ema_change.max(-1).values,
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            # zeroed placeholders (entry caches these; never used in compute)
            "gru_input":   torch.zeros(B, self.gru_input_dim, device=s_t.device),
            "gru_state_post": self.gru_state.detach().clone(),
        }

    # ─── replay (PPO update; gradients flow through reader + fuse) ─────────

    def replay(self, cached_buffer: dict,
               cls_base: torch.Tensor, cls_hand: torch.Tensor,
               proprio: torch.Tensor,
               gru_state_pre: torch.Tensor,     # ignored (API parity)
               gru_input: torch.Tensor) -> torch.Tensor:  # ignored (API parity)
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

    # ─── episode boundary + snapshot/restore (PPO infra) ───────────────────

    def reset(self, env_done_mask: torch.Tensor) -> None:
        self.buffer.reset(env_done_mask)
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if env_done_mask.any():
            self.p_prev[env_done_mask] = 0.0
            self.ema_change[env_done_mask] = 0.0
            self.ms_initialized[env_done_mask] = False

    def snapshot(self) -> dict:
        return {
            "buffer":      self.buffer.state_dict_buffer(),
            "gru_state":   self.gru_state.clone(),   # zeros (API parity)
            "p_prev":      self.p_prev.clone(),
            "ema_change":  self.ema_change.clone(),
            "ms_init":     self.ms_initialized.clone(),
        }

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd["buffer"])
        if "p_prev" in sd:                       # backwards compat
            self.p_prev.copy_(sd["p_prev"])
            self.ema_change.copy_(sd["ema_change"])
            self.ms_initialized.copy_(sd["ms_init"])
