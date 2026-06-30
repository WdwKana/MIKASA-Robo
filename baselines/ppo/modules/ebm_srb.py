"""
EBM-SRB: Self-Referential Episodic Memory.

The buffer is its own surprise oracle. There is NO trained saliency head,
NO trained predictor, NO external attention module.

Per step t:
  p_t = DINOv2(rgb6)                                  (B, N, d_vit) frozen
  surprise[n] = min over m in buffer of ‖p_t[n] − buffer[m]‖²
                 (if buffer empty: fall back to ‖p_t[n]‖²)
  top-K-by-surprise → push to buffer (priority = surprise; FIFO/age eviction)
  LSTM working memory accumulates context
  reader cross-attends to buffer; fuse → s_t → policy

Mechanism story: "memory writes are gated by memory's own retrieval. A new
observation is stored iff no existing entry can act as its near neighbour."
This is the V1 EBM architecture with the saliency_head completely removed
and replaced by a parameter-free retrieval-failure score.

API mirrors `EBMHybridLSTMMemoryModule` so the PPO entry stays diff-minimal.
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


class EBMSRBMemoryModule(nn.Module):
    """Self-Referential Buffer + LSTM working memory + V1-style buffer/reader/fuse.

    The keyword `saliency_ckpt` is accepted for API parity with the other EBM
    modules but is IGNORED — SRB needs no saliency head.
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
    ):
        super().__init__()
        self.num_envs = num_envs
        self.K = K
        self.L = L
        self.d_state = d_state
        self.device = torch.device(device)
        self.vit_backbone = vit_backbone
        self.gru_hidden_size = gru_hidden_size
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

        # 2. Self-referential surprise: min L2 distance to current buffer entries.
        with torch.no_grad():
            buf = self.buffer.features                   # (B, L, d), pre-push
            mask = self.buffer.mask                      # (B, L) bool, true=valid
            any_used = mask.any(dim=-1)                  # (B,)

            # pairwise dist (B, N, L)
            dist = _pairwise_sq_dist(tokens_all, buf)
            # mask invalid entries with +inf so they're never the "nearest"
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
            min_dist = dist.min(dim=-1).values           # (B, N)

            # fallback for envs whose buffer is empty: use feature norm so
            # we still seed the buffer with the most distinctive patches.
            fallback = (tokens_all * tokens_all).sum(-1)
            surprise = torch.where(any_used.unsqueeze(-1), min_dist, fallback)

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

    def snapshot(self) -> dict:
        return {
            "buffer":    self.buffer.state_dict_buffer(),
            "gru_state": self.gru_state.clone(),
        }

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd["buffer"])
        self.gru_state.copy_(sd["gru_state"])
