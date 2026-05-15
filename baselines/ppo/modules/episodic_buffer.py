"""
Episodic K-V buffer with per-environment state.

Holds, for each of the `num_envs` parallel envs, up to `L` token-feature
entries with metadata (timestamp, saliency score). Provides three operations
used by EBM-Robo:

  push_batch(features, saliencies, timestamps, novelty_thresh):
      For each env in the batch, push candidate features that pass a
      cosine-novelty filter against existing buffer contents. Excess entries
      are evicted by lowest priority = saliency * exp(-age / tau_age).

  reset(env_done_mask):
      Wipe buffer state for finished envs.

  get():
      Return stacked (features, mask, timestamps, saliencies) suitable for
      cross-attention.

Stored tensors are pre-allocated for the full buffer size; a per-env
"used" counter tracks the actual fill level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EpisodicBuffer:
    def __init__(self, num_envs: int, L: int, d: int, device,
                 tau_age: float = 30.0, novelty_thresh: float = 0.95):
        self.num_envs = num_envs
        self.L = L
        self.d = d
        self.device = device
        self.tau_age = tau_age
        self.novelty_thresh = novelty_thresh

        self.features  = torch.zeros(num_envs, L, d, device=device)
        self.timestamps = torch.zeros(num_envs, L, dtype=torch.long, device=device)
        self.saliency  = torch.full((num_envs, L), -1e9, device=device)
        self.used      = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def mask(self) -> torch.Tensor:
        """(num_envs, L) bool, True for valid entries."""
        idx = torch.arange(self.L, device=self.device).unsqueeze(0)
        return idx < self.used.unsqueeze(1)

    def reset(self, env_done_mask: torch.Tensor) -> None:
        """env_done_mask: (num_envs,) bool. Reset buffers for True envs."""
        if env_done_mask.dtype != torch.bool:
            env_done_mask = env_done_mask.bool()
        if not env_done_mask.any():
            return
        self.features[env_done_mask] = 0.0
        self.timestamps[env_done_mask] = 0
        self.saliency[env_done_mask] = -1e9
        self.used[env_done_mask] = 0

    @torch.no_grad()
    def push_batch(self, cand_features: torch.Tensor,
                         cand_saliency: torch.Tensor,
                         t_now: int) -> None:
        """For each env, try pushing the K candidate tokens.
        cand_features: (num_envs, K, d)
        cand_saliency: (num_envs, K)  (only positive saliency tokens are pushed)
        t_now: current timestep (scalar).
        """
        B, K, d = cand_features.shape
        feats = F.normalize(cand_features, dim=-1)
        for b in range(B):
            n = int(self.used[b].item())
            buf_feats = self.features[b, :n] if n > 0 else None
            for k in range(K):
                s = float(cand_saliency[b, k].item())
                if s <= 0.0:
                    continue  # below threshold, skip
                v = feats[b, k]
                # novelty check
                if n > 0:
                    sim = (F.normalize(buf_feats, dim=-1) * v).sum(-1).max().item()
                    if sim > self.novelty_thresh:
                        continue
                # push
                if n < self.L:
                    self.features[b, n] = cand_features[b, k]
                    self.timestamps[b, n] = t_now
                    self.saliency[b, n] = s
                    self.used[b] = n + 1
                    n += 1
                    buf_feats = self.features[b, :n]
                else:
                    # evict lowest priority
                    age = (t_now - self.timestamps[b, :n]).float()
                    priority = self.saliency[b, :n] * torch.exp(-age / self.tau_age)
                    if s <= priority.min().item():
                        continue
                    j = int(priority.argmin().item())
                    self.features[b, j] = cand_features[b, k]
                    self.timestamps[b, j] = t_now
                    self.saliency[b, j] = s
                    buf_feats = self.features[b, :n]

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns DETACHED CLONES — buffer values are constants from the
        policy's perspective (they came from frozen ViT under no_grad). Cloning
        prevents subsequent in-place push_batch / reset operations from
        invalidating any autograd graph that referenced them.
          features:   (num_envs, L, d)
          mask:       (num_envs, L) bool
          timestamps: (num_envs, L) long
          saliency:   (num_envs, L)
        """
        return (self.features.detach().clone(),
                self.mask,
                self.timestamps.clone(),
                self.saliency.detach().clone())

    def state_dict_buffer(self) -> dict:
        """Snapshot buffer tensors for PPO replay across rollout."""
        return {
            "features":   self.features.clone(),
            "timestamps": self.timestamps.clone(),
            "saliency":   self.saliency.clone(),
            "used":       self.used.clone(),
        }

    def load_state_dict_buffer(self, sd: dict) -> None:
        self.features.copy_(sd["features"])
        self.timestamps.copy_(sd["timestamps"])
        self.saliency.copy_(sd["saliency"])
        self.used.copy_(sd["used"])
