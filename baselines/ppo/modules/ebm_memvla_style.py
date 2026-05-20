"""
MemoryVLA-style memory module adapted to PPO + DINOv2 single-stream setting.

This is a faithful re-implementation of the memory mechanism from MemoryVLA
(arXiv 2508.19236), stripped of the language path (cognitive stream) so it
runs in our pure-vision online-RL setting. We keep their three distinctive
design choices:

  1. NO write filter — every patch token of every step is pushed.
  2. Cosine-similarity nearest-pair averaging when bank exceeds capacity L.
  3. Two-layer cross-attention readout with timestep positional encoding.

Compared to our EBMMemoryModule (single-stream V1), differences are:
  - Write: push all 162 patches/step (vs our K=8 saliency-filtered).
  - Overflow: merge nearest pair by averaging (vs our priority eviction).
  - Reader: 2 attention layers (vs our 1), no saliency-bias term.
  - No saliency head invoked.

Used as a direct architectural baseline ("MemoryVLA-style memory in our
setting") for Section X of the paper, addressing the question:
"is the perceptual memory architecture of MemoryVLA sufficient for online
memory-RL, or is our explicit write filter necessary?"
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_vit import FrozenDualDinoV2
from .frozen_clip import FrozenDualClip
from .memory_reader import TimestampEmbedding


# ─── Bank ──────────────────────────────────────────────────────────────────

class MemVLABank:
    """Push-all + cosine-pair-merge memory bank, faithful to MemoryVLA.

    Bank shape: (num_envs, L, d). On push of N tokens per env:
      1. Concat new tokens → (used+N) entries
      2. Iteratively find argmax cosine-sim pair and average until size <= L
    """

    def __init__(self, num_envs: int, L: int, d: int, device):
        self.num_envs = num_envs
        self.L = L
        self.d = d
        self.device = device
        self.features = torch.zeros(num_envs, L, d, device=device)
        self.timestamps = torch.zeros(num_envs, L, dtype=torch.long, device=device)
        self.used = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def mask(self) -> torch.Tensor:
        idx = torch.arange(self.L, device=self.device).unsqueeze(0)
        return idx < self.used.unsqueeze(1)

    @torch.no_grad()
    def push_all_with_merge(self, tokens: torch.Tensor, t_now: int) -> None:
        """tokens: (B, N, d) — push all N tokens per env, merge to L.

        Vectorized across envs: build a (B, M, d) padded buffer of all valid
        slots, then in each iteration compute (B, M, M) cosine-sim, pick the
        argmax pair per env from the upper triangle, average that pair, and
        mark the second slot invalid. Repeat until every env's valid count
        is <= self.L. Iterations are sequential (because each merge depends
        on the previous one for that env), but each iteration runs in one
        GPU op for all envs.

        Faithful to MemoryVLA semantics: cosine-similarity-nearest pair is
        averaged (features and timestamps), and the merge is repeated until
        bank size <= L.
        """
        B, N, d = tokens.shape
        L = self.L
        device = self.device

        # Build padded combined: (B, L+N, d) — first L slots from current
        # bank, then N new tokens. valid_mask tells which slots are real.
        M = L + N
        combined = torch.empty(B, M, d, device=device)
        combined[:, :L] = self.features
        combined[:, L:] = tokens
        ts = torch.empty(B, M, dtype=torch.long, device=device)
        ts[:, :L] = self.timestamps
        ts[:, L:] = t_now
        valid_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        l_idx = torch.arange(L, device=device).unsqueeze(0)
        valid_mask[:, :L] = l_idx < self.used.unsqueeze(1)
        valid_mask[:, L:] = True

        batch_idx = torch.arange(B, device=device)
        tri = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool),
                         diagonal=1).unsqueeze(0)  # (1, M, M)

        # Batch K disjoint pair-merges per outer iteration. With N=162 and
        # L=64, we need ~162 merges in steady state; with K=16, that's ~10
        # outer iterations. Greedy disjoint-pair selection per env: pick
        # argmax, mask out its row/col, repeat K times. Approximation to
        # strict sequential pair-merge (a merged token's new similarities
        # are only recomputed every K rounds), but preserves MemoryVLA's
        # "merge closest pair by averaging" semantics.
        BATCH_K = 16
        for _ in range(M // BATCH_K + 2):  # safety bound
            counts = valid_mask.sum(dim=-1)
            if counts.max().item() <= L:
                break
            # need_k[b] = how many merges env b still needs (0 if done)
            need_k = (counts - L).clamp_(min=0)

            norm = F.normalize(combined, dim=-1)
            sim = torch.bmm(norm, norm.transpose(1, 2))  # (B, M, M)
            pair_valid = tri & valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            sim = sim.masked_fill(~pair_valid, float("-inf"))

            # Greedy disjoint selection: K rounds of argmax-then-mask.
            picked_i = []
            picked_j = []
            for r in range(BATCH_K):
                flat = sim.view(B, -1).argmax(dim=-1)
                i_idx = (flat // M).long()
                j_idx = (flat %  M).long()
                # mask row/col of i and j for next round to keep disjointness
                row_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
                row_mask[batch_idx, i_idx] = True
                row_mask[batch_idx, j_idx] = True
                # broadcast over rows and cols
                sim.masked_fill_(row_mask.unsqueeze(2), float("-inf"))
                sim.masked_fill_(row_mask.unsqueeze(1), float("-inf"))
                picked_i.append(i_idx)
                picked_j.append(j_idx)
            picked_i = torch.stack(picked_i, dim=1)  # (B, K)
            picked_j = torch.stack(picked_j, dim=1)  # (B, K)

            # Apply merges: for each (b, r), only if r < need_k[b].
            arange_k = torch.arange(BATCH_K, device=device).unsqueeze(0)  # (1, K)
            do_merge = arange_k < need_k.unsqueeze(1)  # (B, K)

            # Flatten env+round so we can use scatter / gather.
            b_flat = batch_idx.unsqueeze(1).expand(B, BATCH_K)  # (B, K)
            i_f = combined[b_flat.reshape(-1), picked_i.reshape(-1)]  # (B*K, d)
            j_f = combined[b_flat.reshape(-1), picked_j.reshape(-1)]
            i_t = ts[b_flat.reshape(-1), picked_i.reshape(-1)]
            j_t = ts[b_flat.reshape(-1), picked_j.reshape(-1)]
            merged_f = 0.5 * (i_f + j_f)
            merged_t = (i_t + j_t) // 2
            do_flat = do_merge.reshape(-1)

            # write merged into i, keep original where not merging
            new_i_feat = torch.where(do_flat.unsqueeze(-1), merged_f, i_f)
            new_i_ts = torch.where(do_flat, merged_t, i_t)
            combined[b_flat.reshape(-1), picked_i.reshape(-1)] = new_i_feat
            ts[b_flat.reshape(-1), picked_i.reshape(-1)] = new_i_ts

            # drop slot j where merging happened
            cur_v = valid_mask[b_flat.reshape(-1), picked_j.reshape(-1)]
            valid_mask[b_flat.reshape(-1), picked_j.reshape(-1)] = cur_v & (~do_flat)

        # Compact: for each env, gather the valid slots to the front. Counts
        # are now all <= L; pad to L with zeros.
        # We use sort: invalid slots get a high key so they fall to the back.
        sort_key = (~valid_mask).long()  # 0 for valid, 1 for invalid
        order = sort_key.argsort(dim=-1, stable=True)  # (B, M); valids first
        # Gather first L
        order_L = order[:, :L]  # (B, L)
        gather_idx_feat = order_L.unsqueeze(-1).expand(-1, -1, d)
        new_features = torch.gather(combined, 1, gather_idx_feat)  # (B, L, d)
        new_timestamps = torch.gather(ts, 1, order_L)               # (B, L)
        new_valid = torch.gather(valid_mask, 1, order_L)            # (B, L)
        # Zero out positions past the valid count (defensive; argsort with
        # stable=True keeps valids first so this is for trailing pad slots).
        new_features = new_features * new_valid.unsqueeze(-1).float()
        new_timestamps = new_timestamps * new_valid.long()

        self.features.copy_(new_features)
        self.timestamps.copy_(new_timestamps)
        self.used.copy_(new_valid.sum(dim=-1).long().clamp_(max=L))

    def get(self):
        return (
            self.features.detach().clone(),
            self.mask,
            self.timestamps.clone(),
        )

    def reset(self, env_done_mask: torch.Tensor) -> None:
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if env_done_mask.any():
            self.features[env_done_mask] = 0.0
            self.timestamps[env_done_mask] = 0
            self.used[env_done_mask] = 0

    def state_dict_buffer(self) -> dict:
        return {
            "features": self.features.clone(),
            "timestamps": self.timestamps.clone(),
            "used": self.used.clone(),
        }

    def load_state_dict_buffer(self, sd: dict) -> None:
        self.features.copy_(sd["features"])
        self.timestamps.copy_(sd["timestamps"])
        self.used.copy_(sd["used"])


# ─── Two-layer cross-attention reader (MemoryVLA uses 2 layers) ───────────

class TwoLayerMemVLAReader(nn.Module):
    """Mirrors MemoryVLA's two-layer cross-attention; query is current cls
    summary + proprio (their VLA uses perceptual + cognitive tokens but we
    don't have language)."""

    def __init__(self, d_query: int, d_buffer: int, d_proj: int = 256):
        super().__init__()
        self.d_proj = d_proj
        # Query projection
        self.W_Q = nn.Linear(d_query, d_proj)
        # Per-layer K/V projections
        self.W_K1 = nn.Linear(d_buffer, d_proj)
        self.W_V1 = nn.Linear(d_buffer, d_proj)
        self.W_K2 = nn.Linear(d_proj, d_proj)
        self.W_V2 = nn.Linear(d_proj, d_proj)
        # Layer 2 takes layer-1 output as query
        self.ts_embed = TimestampEmbedding(d_proj)
        self.out_norm = nn.LayerNorm(d_proj)

    def forward(self, query_in, buffer_features, buffer_mask, buffer_timestamps):
        B, L, _ = buffer_features.shape
        Q = self.W_Q(query_in)                                  # (B, d_proj)
        ts_e = self.ts_embed(buffer_timestamps)                 # (B, L, d_proj)

        # Layer 1
        K1 = self.W_K1(buffer_features) + ts_e
        V1 = self.W_V1(buffer_features)
        scale = self.d_proj ** 0.5
        logits1 = torch.einsum("bd,bld->bl", Q, K1) / scale
        logits1 = logits1.masked_fill(~buffer_mask, -1e9)
        any_valid = buffer_mask.any(dim=-1, keepdim=True)
        attn1 = logits1.softmax(dim=-1)
        out1 = torch.einsum("bl,bld->bd", attn1, V1)            # (B, d_proj)
        out1 = out1 * any_valid.float()

        # Layer 2: use out1 as new query
        K2 = self.W_K2(V1) + ts_e                                # re-use V1 as K-source
        V2 = self.W_V2(V1)
        logits2 = torch.einsum("bd,bld->bl", out1, K2) / scale
        logits2 = logits2.masked_fill(~buffer_mask, -1e9)
        attn2 = logits2.softmax(dim=-1)
        out2 = torch.einsum("bl,bld->bd", attn2, V2)
        out2 = out2 * any_valid.float()

        return self.out_norm(out2)


# ─── Full module (parallels EBMMemoryModule API) ──────────────────────────

class EBMMemVLAStyleModule(nn.Module):
    """Drop-in replacement for EBMMemoryModule that uses MemoryVLA's
    write-all + cosine-pair-merge bank + 2-layer cross-attn reader.

    No saliency head is loaded; the perception path is just frozen DINOv2.
    """

    def __init__(
        self,
        num_envs: int,
        proprio_dim: int = 25,
        saliency_ckpt: str | Path | None = None,   # unused; kept for API parity
        vit_backbone: str = "dinov2",
        L: int = 64,
        K: int = 8,                                # unused; kept for API parity
        d_proj: int = 256,
        d_state: int = 256,
        novelty_thresh: float = 0.95,              # unused
        tau_age: float = 30.0,                     # unused
        device: str | torch.device = "cuda",
        no_saliency: bool = False,                 # always True in spirit; unused
    ):
        super().__init__()
        self.num_envs = num_envs
        self.L = L
        self.d_state = d_state
        self.device = torch.device(device)
        self.vit_backbone = vit_backbone

        # Frozen perception
        if vit_backbone == "dinov2":
            self.vit = FrozenDualDinoV2()
        elif vit_backbone == "clip":
            self.vit = FrozenDualClip()
        else:
            raise ValueError(f"unknown vit_backbone: {vit_backbone}")
        self.vit.to(self.device)
        d_vit = self.vit.dim

        # Learned current-frame summary
        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        summary_dim = 128

        # MemoryVLA-style bank
        self.buffer = MemVLABank(
            num_envs=num_envs, L=L, d=d_vit, device=self.device,
        )

        # Two-layer cross-attention reader
        self.reader = TwoLayerMemVLAReader(
            d_query=proprio_dim + summary_dim, d_buffer=d_vit, d_proj=d_proj,
        )

        # Fuse
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + summary_dim + d_proj, 256), nn.GELU(),
            nn.Linear(256, d_state),
        )

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        # 1. Frozen ViT
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)   # (B, 2*N_v, d_vit)

        # 2. Push ALL tokens to bank with cosine-pair merge on overflow
        with torch.no_grad():
            self.buffer.push_all_with_merge(tokens_all, t_now=t)

        # 3. Build query (no language)
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        query_in = torch.cat([proprio, curr], dim=-1)

        # 4. Two-layer cross-attention readout
        feats, mask, ts = self.buffer.get()
        retrieved = self.reader(query_in, feats, mask, ts)

        # 5. Fuse → actor/critic state
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))
        return s_t, {
            "buffer_used": self.buffer.used.clone(),
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
        }

    def replay(self, cached_buffer: dict, cls_base, cls_hand, proprio) -> torch.Tensor:
        """Differentiable replay of cls_summary + reader + fuse on cached bank.
        No GRU here (this is V1-like architecture). No gru_state needed."""
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
        )
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))
        return s_t

    def reset(self, env_done_mask: torch.Tensor) -> None:
        self.buffer.reset(env_done_mask)

    def snapshot(self) -> dict:
        return self.buffer.state_dict_buffer()

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd)
