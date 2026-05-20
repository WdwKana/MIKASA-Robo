"""
EBM-Hybrid-LSTM (V3a) — V2 architecture with the parallel GRU swapped for an
LSTM. Motivation: A6-LSTM is the strongest pure-recurrent baseline we have on
trajectory tasks (Intercept family); swap GRU → LSTM to see whether the gated
cell helps the hybrid too.

Storage trick to keep PPO-entry diff small: we expose state as a single
"flat" tensor of shape (1, B, 2*H) — `[h ; c]` concatenated along the last
dim. The PPO entry caches and replays it identically to the GRU-Hybrid path.
The module splits it back into (h, c) before the LSTM call and re-concats
afterwards.

Everything else (saliency, buffer, reader, fuse) is identical to V2.
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


class EBMHybridLSTMMemoryModule(nn.Module):
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
        gru_hidden_size: int = 128,   # name kept for PPO-entry compat
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
        H = gru_hidden_size

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

        self.reader = MemoryReader(
            d_query=proprio_dim + summary_dim, d_buffer=d_vit, d_proj=d_proj,
        )

        # LSTM input matches V2 GRU: pooled top-K + curr + proprio
        self.gru_input_dim = d_vit + summary_dim + proprio_dim
        self.lstm = nn.LSTM(self.gru_input_dim, H, num_layers=1, batch_first=False)
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)

        # Flat state buffer: [h ; c] concatenated, shape (1, B, 2H)
        self.register_buffer("gru_state",
                             torch.zeros(1, num_envs, 2 * H, device=self.device))

        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj + H, 256),
            nn.GELU(),
            nn.Linear(256, d_state),
        )

    def _split_state(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.gru_hidden_size
        return flat[..., :H].contiguous(), flat[..., H:].contiguous()

    def _join_state(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, c], dim=-1)

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
        pool_w = torch.softmax(cand_sal, dim=-1).unsqueeze(-1)
        pooled = (pool_w * cand_feats).sum(dim=1)

        gru_input = torch.cat([pooled, curr, proprio], dim=-1).unsqueeze(0)
        with torch.no_grad():
            h, c = self._split_state(self.gru_state)
            _, (new_h, new_c) = self.lstm(gru_input, (h, c))
            self.gru_state = self._join_state(new_h, new_c).detach()
        gru_hidden = new_h.squeeze(0)

        feats, mask, ts, sal = self.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(query_in, feats, mask, ts, sal)

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
