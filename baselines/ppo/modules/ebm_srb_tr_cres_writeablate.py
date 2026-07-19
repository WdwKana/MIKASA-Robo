"""
EBM-SRB-TR-CRES-WriteAblate: write-rule ablation variants of the main method.

Answers the reviewer question "does the self-referential surprise gate actually
matter, or does any write rule fill the buffer just as well?" by replacing ONLY
the buffer write rule with content-blind alternatives. Everything else —
frozen ViT, CRES color fusion, motion EMA, the LSTM motion-routing branch,
buffer capacity/eviction machinery, cross-attention reader, fuse — is inherited
unchanged from `ebm_srb_tr_cres.py`.

Two ablation rules (the method's own rule, "surprise", lives ONLY in the main
module/entry — this file deliberately cannot run it, so main-table runs remain
exactly reproducible from the untouched canonical files):

  random : K uniform-random tokens per step; constant write priority (so
           eviction degenerates to age-only); cosine dedup KEPT.
           -> ablates the surprise SCORING alone.
  fifo   : same random selection, but the cosine novelty filter is DISABLED
           (novelty_thresh=2.0 > max cosine) -> fully content-blind ring
           buffer (age-only eviction, no dedup).
           -> ablates ALL content-based writing.

The random->surprise delta isolates the scoring signal; fifo->random isolates
the dedup filter.

step() below is a copy of the CRES step() with ONLY the selection block
(steps 3+5) swapped — kept as a full copy per repo convention (variants are
standalone files; canonical modules are never edited).
"""
from __future__ import annotations

import torch

from .ebm_srb_tr_cres import EBMSRBTRCRESMemoryModule


class EBMSRBTRCRESWriteAblateMemoryModule(EBMSRBTRCRESMemoryModule):
    """SRB-TR-CRES with a content-blind buffer write rule (random | fifo)."""

    def __init__(self, *args, write_rule: str = "", **kwargs):
        if write_rule not in ("random", "fifo"):
            raise ValueError(
                f"write_rule must be 'random' or 'fifo' for the ablation module "
                f"(got {write_rule!r}); the 'surprise' rule is the main method — "
                f"run ppo_memtasks_ebm_srb_tr_cres_caps.py for it.")
        super().__init__(*args, **kwargs)
        self.write_rule = write_rule
        if write_rule == "fifo":
            self.buffer.novelty_thresh = 2.0   # cosine <= 1 always passes -> dedup off

    # ─── per-step forward (copy of CRES step(); selection block swapped) ───
    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int):
        # 1. ViT
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)         # (B, N, d_vit)
        B, N, d = tokens_all.shape

        with torch.no_grad():
            # 2. Color-fused feature used for ALL buffer operations.
            z = self._fuse_color(tokens_all, rgb6)            # (B, N, d_vit)

            # 4. Motion suppression on PURE DINOv2 features (color is static).
            # (kept: the LSTM route below uses ema_change — routing untouched)
            init_now = self.ms_initialized
            cur_change = ((tokens_all - self.p_prev) ** 2).sum(-1)
            upd = self.ms_alpha * cur_change + (1 - self.ms_alpha) * self.ema_change
            self.ema_change = torch.where(init_now.unsqueeze(-1), upd, self.ema_change)
            self.p_prev = tokens_all.detach()
            self.ms_initialized = torch.ones_like(self.ms_initialized)

            # 3+5 ABLATED: content-blind selection — K uniform-random tokens,
            # constant priority (age-only eviction). Same z storage, same
            # capacity, same reader. No surprise computed anywhere.
            rand_score = torch.rand(B, N, device=z.device)
            topk_val_buf, topk_idx_buf = rand_score.topk(self.K, dim=-1)
            cand_feats = z.gather(1, topk_idx_buf.unsqueeze(-1).expand(B, self.K, d))
            cand_priority = torch.ones_like(topk_val_buf)

            # 6. LSTM route: high-motion patches, pooled from PURE features.
            topk_val_lstm, topk_idx_lstm = self.ema_change.topk(self.K, dim=-1)
            lstm_feats = tokens_all.gather(
                1, topk_idx_lstm.unsqueeze(-1).expand(B, self.K, d))

        # 7. Push color-aware features to buffer.
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_priority, t_now=t)
        n_pushed = (cand_priority > 0).sum(dim=-1)

        # 8-10. Rest mirrors SRB-TR-CRES exactly.
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
            "max_surprise": topk_val_buf.max(-1).values,   # rand scores (diagnostic)
            "max_motion":   topk_val_lstm.max(-1).values,
            "color_scale": self.color_scale.clone(),
            "cls_base":    cls_b.detach(),
            "cls_hand":    cls_h.detach(),
            "gru_input":   gru_input.squeeze(0).detach(),
            "gru_state_post": self.gru_state.detach().clone(),
        }
