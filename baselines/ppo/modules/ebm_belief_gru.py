"""
EBM-Belief-GRU (V4-GRU) — Belief-State Episodic Memory.

Distinguishing principle vs. V2/V3: the recurrent state plays TWO functional
roles instead of one.

  1. Direct feed to fuse  → preserves trajectory integration (the V2 win
                            on Intercept tasks; pure GRU-as-query loses this
                            because integrated quantities like velocity are
                            not stored in any single buffer slot).
  2. Query for retrieval  → belief-driven cross-attention; the agent's
                            evolving belief about the task drives what is
                            fetched from past evidence. Replaces V1/V2's
                            static query `[proprio, curr_summary]`.

This is the deep-learning analog of a POMDP belief filter:
  - short-horizon continuous integration  → GRU hidden state b_t
  - long-horizon discrete events          → episodic buffer
  - belief drives retrieval; retrieved evidence informs next belief update.

Architectural differences from V3b (Clean-GRU Hybrid):
  - GRU input is `[proprio, curr]` (clean, like V3b).
  - Cross-attention reader query is `b_t` (gru_hidden), not `[proprio, curr]`.
  - GRU hidden also flows directly into fuse (dual-use).

Module API matches V2/V3 (step / replay / reset / snapshot / restore) so the
PPO entry diff stays small.
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


class EBMBeliefGRUMemoryModule(nn.Module):
    def __init__(
        self,
        num_envs: int,
        proprio_dim: int = 25,
        saliency_ckpt: str | Path | None = None,
        vit_backbone: str = "dinov2",
        L: int = 64,
        K: int = 8,
        d_proj: int = 256,
        d_state: int = 256,
        novelty_thresh: float = 0.95,
        tau_age: float = 30.0,
        gru_hidden_size: int = 128,
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
        self.gru_hidden_size = gru_hidden_size

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

        if saliency_ckpt is None:
            raise ValueError("saliency_ckpt is required.")
        self.saliency_head, xy_concat = load_saliency_head(
            saliency_ckpt, device=self.device, freeze=True)
        self.register_buffer("xy_concat", xy_concat)

        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        summary_dim = 128

        self.buffer = EpisodicBuffer(
            num_envs=num_envs, L=L, d=d_vit, device=self.device,
            tau_age=tau_age, novelty_thresh=novelty_thresh,
        )

        # READER: query dim = gru_hidden_size (belief drives retrieval).
        self.reader = MemoryReader(
            d_query=gru_hidden_size, d_buffer=d_vit, d_proj=d_proj,
        )

        # GRU input: clean (proprio + curr).
        self.gru_input_dim = summary_dim + proprio_dim
        self.gru = nn.GRU(self.gru_input_dim, gru_hidden_size, num_layers=1, batch_first=False)
        for name, p in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)
        self.register_buffer("gru_state",
                             torch.zeros(1, num_envs, gru_hidden_size, device=self.device))

        # Fuse takes proprio + curr + retrieved + gru_hidden (dual-use).
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj + gru_hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, d_state),
        )

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)

        with torch.no_grad():
            B = tokens_all.size(0)
            if self.no_saliency:
                idx = torch.randint(0, tokens_all.size(1), (B, self.K), device=tokens_all.device)
                cand_feats = tokens_all.gather(
                    1, idx.unsqueeze(-1).expand(B, self.K, tokens_all.size(-1)))
                cand_sal = torch.ones(B, self.K, device=tokens_all.device)
                probs = torch.zeros(B, tokens_all.size(1), device=tokens_all.device)
            else:
                logits = self.saliency_head(tokens_all, self.xy_concat)
                probs = torch.sigmoid(logits)
                topk_val, topk_idx = probs.topk(self.K, dim=-1)
                cand_feats = tokens_all.gather(
                    1, topk_idx.unsqueeze(-1).expand(B, self.K, tokens_all.size(-1)))
                cand_sal = topk_val * (topk_val > 0.5).float()

        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_sal, t_now=t)
        n_pushed = (cand_sal > 0).sum(dim=-1)

        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))

        # 1. Belief update (GRU) — clean input.
        gru_input = torch.cat([proprio, curr], dim=-1).unsqueeze(0)
        with torch.no_grad():
            new_gru_state, _ = self.gru(gru_input, self.gru_state)
            self.gru_state = new_gru_state.detach()
        gru_hidden = self.gru_state.squeeze(0)

        # 2. Belief-driven retrieval (query = gru_hidden).
        feats, mask, ts, sal = self.buffer.get()
        retrieved = self.reader(gru_hidden, feats, mask, ts, sal)

        # 3. Fuse: dual-use of belief (both as query above AND direct here).
        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "saliency_max": probs.max(-1).values,
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),
            "gru_state_post": self.gru_state.detach().clone(),
        }

    def replay(self, cached_buffer: dict,
               cls_base: torch.Tensor, cls_hand: torch.Tensor,
               proprio: torch.Tensor,
               gru_state_pre: torch.Tensor,
               gru_input: torch.Tensor) -> torch.Tensor:
        L = cached_buffer["features"].shape[1]
        idx = torch.arange(L, device=cached_buffer["features"].device).unsqueeze(0)
        mask = idx < cached_buffer["used"].unsqueeze(1)

        curr = self.curr_summary(torch.cat([cls_base, cls_hand], dim=-1))

        # Belief update — differentiable through GRU.
        new_gru_state, _ = self.gru(gru_input.unsqueeze(0), gru_state_pre)
        gru_hidden = new_gru_state.squeeze(0)

        # Belief-driven retrieval (gradient flows back into GRU through query).
        retrieved = self.reader(
            gru_hidden,
            cached_buffer["features"],
            mask,
            cached_buffer["timestamps"],
            cached_buffer["saliency"],
        )

        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))
        return s_t

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
