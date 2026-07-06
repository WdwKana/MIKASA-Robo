"""GRU-interface adapters for vendored recurrent baselines (FFM, SHM).

Our recurrent PPO entries (see ppo_memtasks_dinov2_gru*.py) plumb hidden state as
a real-valued tensor shaped (num_layers, B, hidden_size): zero-initialised at
episode start, reset by `(1 - done) * state`, cached per-step during rollout and
replayed per-sample in flat minibatches. These adapters expose the vendored
cells through exactly that interface so the PPO entry code is unchanged except
for the module swap — which keeps the comparison attributable to the memory
cell, not to harness differences.

Both vendored cells have all-zeros initial states, so zeros-init and
`(1 - done) * state` masking are exact episode resets for them too.

Adapter contract (mirrors the parts of nn.GRU the entries use):
  attrs:   num_layers=1, input_size, hidden_size (= FLAT STATE SIZE, used for
           state-buffer allocation), output_size (= feature size fed to heads;
           for nn.GRU these coincide, here they may differ)
  forward: (x: (1, B, input_size), state: (1, B, hidden_size))
           -> (y: (1, B, output_size), state: (1, B, hidden_size))
  Only single-step (T=1) calls are used by the entries (rollout steps and
  flat-minibatch replay with cached pre-states).
"""
from __future__ import annotations

import torch
from torch import nn

from .ffm_vendored import DropInFFM
from .shm_vendored import SHMCell


class FFMRecurrentAdapter(nn.Module):
    """DropInFFM behind the GRU state interface.

    FFM's recurrent state is complex64 (B, memory_size, context_size); we store
    it flat as float32 (1, B, 2*m*c) via view_as_real. Scaling the real view by
    (1 - done) scales the complex state identically, so the harness's reset
    masking is exact.
    """

    def __init__(self, input_size: int, memory_size: int = 64,
                 context_size: int = 4, output_size: int = 512):
        super().__init__()
        self.core = DropInFFM(
            input_size=input_size,
            hidden_size=output_size,   # stored by FFM but unused in its math
            memory_size=memory_size,
            context_size=context_size,
            output_size=output_size,
            batch_first=True,
        )
        self.num_layers = 1
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.hidden_size = 2 * memory_size * context_size  # flat float state

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        # x: (1, B, F) seq-first; state: (1, B, 2*m*c) float32
        B = x.shape[1]
        s = state.squeeze(0).reshape(B, self.memory_size, self.context_size, 2)
        s = torch.view_as_complex(s.contiguous())
        y, s = self.core(x.squeeze(0), s)          # (B, F) -> (B, out), (B, m, c)
        s = torch.view_as_real(s).reshape(1, B, self.hidden_size)
        return y.unsqueeze(0), s


class SHMRecurrentAdapter(nn.Module):
    """SHMCell behind the GRU state interface.

    SHM's state is the (B, H, H) memory matrix (H = mem_size), stored flat as
    (1, B, H*H). post_size is the cell's read-out width and becomes the feature
    size fed to the actor/critic heads.
    """

    def __init__(self, input_size: int, mem_size: int = 72, post_size: int = 1024):
        super().__init__()
        self.core = SHMCell(input_size=input_size, mem_size=mem_size,
                            post_size=post_size)
        self.num_layers = 1
        self.input_size = input_size
        self.output_size = post_size
        self.mem_size = mem_size
        self.hidden_size = mem_size * mem_size  # flat float state

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        # x: (1, B, F) seq-first; state: (1, B, H*H) float32
        B = x.shape[1]
        s = state.squeeze(0).reshape(B, self.mem_size, self.mem_size)
        y, s = self.core(x.permute(1, 0, 2), s)    # (B, 1, F) -> (B, 1, out), (B, H, H)
        return y.permute(1, 0, 2), s.reshape(1, B, self.hidden_size)
