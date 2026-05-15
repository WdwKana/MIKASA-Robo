"""
EBM-Hybrid Memory Module (V2 for trajectory tasks).

Composition of V1 EBM components + a small GRU(128) running in parallel.

Rationale: V1 EBM (Buffer + cross-attn) excels on episodic memory but is
weak on trajectory tasks (InterceptMedium, InterceptGrab) where the agent
needs to integrate motion continuously. RNNs naturally fit motion integration.
Adding a small GRU branch alongside the buffer gives the agent BOTH a discrete
episodic memory (buffer) AND a continuous motion integrator (GRU).

Architecture per env step:

  rgb6 (B,128,128,6), proprio (B,25)
        │
        ▼  Frozen DINOv2-S/14 (or CLIP via vit_backbone arg)
  tok_b, tok_h, cls_b, cls_h
        │
        ▼  Frozen saliency head v3
  top-K=8 patches → cand_feats (B,K,d_vit)
        │
        ├──── push to Buffer ────► cross-attn read ────► retrieved (B,d_proj)
        │
        ▼  saliency-weighted pool over top-K
  pooled (B, d_vit)
        │
  curr_summary([cls_b, cls_h])  ────► curr (B, 128)
        │
        ▼
  gru_input = concat([pooled, curr, proprio])  ────► GRU(input, gru_state)
                                                          │
                                                          ▼
                                                     gru_hidden (B, 128)
        │
        ▼  MLP_fuse([proprio, curr, retrieved, gru_hidden])  ────► s_t (B, d_state)

Memory in this module:
  - per-env Buffer state (V1, K-V tokens with timestamps + saliency)
  - per-env GRU hidden state (V2, dense vector for motion integration)
  Both reset together when an env's episode ends.
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


class EBMHybridMemoryModule(nn.Module):
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

        # --- frozen perception ----------------------------------------------
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

        # --- frozen saliency head ------------------------------------------
        if saliency_ckpt is None:
            raise ValueError("saliency_ckpt is required.")
        self.saliency_head, xy_concat = load_saliency_head(
            saliency_ckpt, device=self.device, freeze=True)
        self.register_buffer("xy_concat", xy_concat)

        # --- learned: current-frame summary --------------------------------
        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        summary_dim = 128

        # --- buffer (per-env, non-trainable state) -------------------------
        self.buffer = EpisodicBuffer(
            num_envs=num_envs, L=L, d=d_vit, device=self.device,
            tau_age=tau_age, novelty_thresh=novelty_thresh,
        )

        # --- learned: memory reader (cross-attn) ---------------------------
        self.reader = MemoryReader(
            d_query=proprio_dim + summary_dim, d_buffer=d_vit, d_proj=d_proj,
        )

        # --- NEW: per-env GRU branch for trajectory integration ------------
        # GRU input is the SAME information that flows through the buffer/reader
        # path: saliency-weighted pool + current cls summary + proprio. This
        # ensures the GRU sees per-step task-relevant content (not pixel noise).
        self.gru_input_dim = d_vit + summary_dim + proprio_dim
        self.gru = nn.GRU(self.gru_input_dim, gru_hidden_size, num_layers=1, batch_first=False)
        for name, p in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)
        # per-env hidden state (1, B, H); managed internally
        self.register_buffer("gru_state",
                             torch.zeros(1, num_envs, gru_hidden_size, device=self.device))

        # --- learned: fuse layer (now also takes gru_hidden) ---------------
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj + gru_hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, d_state),
        )

    # ─── per-step forward ───────────────────────────────────────────────────

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        # 1. ViT (no_grad)
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)

        # 2. Saliency head (frozen)
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

        # 3. Push to buffer (V1 path)
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_sal, t_now=t)
        n_pushed = (cand_sal > 0).sum(dim=-1)

        # 4. curr_summary + saliency-weighted pool of top-K (V2 path content)
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        # softmax-weighted pool over K candidates (same content the reader effectively reads)
        pool_w = torch.softmax(cand_sal, dim=-1).unsqueeze(-1)            # (B, K, 1)
        pooled = (pool_w * cand_feats).sum(dim=1)                           # (B, d_vit)

        # 5. GRU step (per-env hidden state managed internally; no grad here)
        gru_input = torch.cat([pooled, curr, proprio], dim=-1).unsqueeze(0)  # (1, B, gru_in)
        with torch.no_grad():
            new_gru_state, _ = self.gru(gru_input, self.gru_state)
            self.gru_state = new_gru_state.detach()
        gru_hidden = self.gru_state.squeeze(0)                              # (B, gru_hidden)

        # 6. Cross-attn read
        feats, mask, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask, ts, sal)             # (B, d_proj)

        # 7. Fuse — actor/critic state includes BOTH buffer-retrieved + GRU-integrated
        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "saliency_max": probs.max(-1).values,
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),  # for replay caching
            "gru_state_post": self.gru_state.detach().clone(),
        }

    # ─── replay (used during PPO update, differentiable through GRU+reader) ─

    def replay(self, cached_buffer: dict,
               cls_base: torch.Tensor, cls_hand: torch.Tensor,
               proprio: torch.Tensor,
               gru_state_pre: torch.Tensor,
               gru_input: torch.Tensor) -> torch.Tensor:
        """
        Inputs are the PRE-STEP gru_state and the recorded gru_input for that
        step. Forward-passes GRU + reader + fuse so gradient flows through
        all learned modules.

        gru_state_pre: (1, mb_size, gru_hidden)
        gru_input:     (mb_size, gru_input_dim)
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

        # GRU 1-step forward, differentiable through gru params
        new_gru_state, _ = self.gru(gru_input.unsqueeze(0), gru_state_pre)
        gru_hidden = new_gru_state.squeeze(0)

        s_t = self.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))
        return s_t

    # ─── episode boundary ──────────────────────────────────────────────────

    def reset(self, env_done_mask: torch.Tensor) -> None:
        self.buffer.reset(env_done_mask)
        # zero out GRU hidden for envs that just terminated
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if env_done_mask.any():
            self.gru_state[:, env_done_mask] = 0.0

    # ─── PPO buffer snapshot/restore (for bootstrap value calls) ───────────

    def snapshot(self) -> dict:
        return {
            "buffer":    self.buffer.state_dict_buffer(),
            "gru_state": self.gru_state.clone(),
        }

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd["buffer"])
        self.gru_state.copy_(sd["gru_state"])
