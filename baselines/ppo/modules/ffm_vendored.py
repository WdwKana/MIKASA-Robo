"""VENDORED baseline code — Fast and Forgetful Memory (FFM).

Source: https://github.com/proroklab/ffm  (MIT License, Copyright (c) 2023)
Paper:  "Reinforcement Learning with Fast and Forgetful Memory",
        Morad et al., NeurIPS 2023, arXiv:2310.04128.

Files `standalone/ffm/ffa.py` and `standalone/ffm/ffm.py` copied VERBATIM from the
authors' standalone package (the distribution the authors provide precisely for
drop-in use), joined into one module (the only change: the package-relative
`from .ffa import FFA, LogspaceFFA` is resolved by concatenation). Do NOT modify
the math here — any adaptation to our PPO harness lives in
`recurrent_adapters.py`, so that baseline results are attributable to the
authors' implementation, not to our re-implementation choices.
"""
# flake8: noqa
from typing import Optional, Tuple
import math
import torch
from torch import nn


# ────────────────────────────────────────────────────────────────────────────
# standalone/ffm/ffa.py (verbatim)
# ────────────────────────────────────────────────────────────────────────────
class FFA(nn.Module):
    def __init__(
        self,
        # Required
        memory_size: int,
        context_size: int,
        # Model Settings
        max_len: int = 1024,
        dtype: torch.dtype = torch.double,
        oscillate: bool = True,
        learn_oscillate: bool = True,
        decay: bool = True,
        learn_decay: bool = True,
        fudge_factor: float = 0.01,
        # Weight Init Settings
        min_period: int = 1,
        max_period: int = 1024,
        forgotten_at: float = 0.01,
        modify_real: bool = True,
    ):
        """A phaser-encoded aggregation operator"""
        super().__init__()
        self.memory_size = memory_size
        self.max_len = max_len
        self.context_size = context_size
        self.oscillate = oscillate
        self.learn_oscillate = learn_oscillate
        self.decay = decay
        self.learn_decay = learn_decay
        assert dtype in [torch.float, torch.double]
        self.dtype = dtype
        dtype_max = torch.finfo(dtype).max
        self.limit = math.log(dtype_max) / max_len - fudge_factor

        if modify_real:
            soft_high = math.log(forgotten_at) / max_period
        else:
            soft_high = -1e-6

        real_param_shape = [1, 1, self.memory_size]
        imag_param_shape = [1, 1, self.context_size]
        a_low = -self.limit + fudge_factor
        a_high = max(min(-1e-6, soft_high), a_low)

        a = torch.linspace(a_low, a_high, real_param_shape[-1]).reshape(
            real_param_shape
        )
        b = (
            2
            * torch.pi
            / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(
                imag_param_shape
            )
        )

        if not self.oscillate:
            b.fill_(1 / 1e6)
        if not self.decay:
            a.fill_(0)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # For typechecker
        self.one: torch.Tensor
        self.inner_idx: torch.Tensor
        self.outer_idx: torch.Tensor
        self.state_offset: torch.Tensor
        # Buffers
        self.register_buffer("one", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("inner_idx", torch.arange(max_len, dtype=dtype).flip(0))
        self.register_buffer("outer_idx", -self.inner_idx)
        self.register_buffer("state_offset", torch.arange(1, max_len + 1, dtype=dtype))

    def extra_repr(self):
        return f"in_features={self.memory_size}, out_features=({self.memory_size}, {self.context_size})"

    def log_gamma(self, t_minus_i: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        assert t_minus_i.dim() == 1
        T = t_minus_i.shape[0]
        if clamp:
            self.a.data = self.a.data.clamp(min=-self.limit, max=-1e-8)
            a = self.a.clamp(min=-self.limit, max=-1e-8)
        else:
            a = self.a
        b = self.b
        if not self.oscillate or not self.learn_oscillate:
            b = b.detach()
        if not self.decay or not self.learn_decay:
            a = a.detach()

        exp = torch.complex(
            a.reshape(1, 1, -1, 1),
            b.reshape(1, 1, 1, -1),
        )
        out = exp * t_minus_i.reshape(1, T, 1, 1)
        return out

    def gamma(self, t_minus_i: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return self.log_gamma(t_minus_i, clamp=clamp).exp()

    def compute_z(self, x):
        B, T, F, D = x.shape
        return torch.cumsum(self.gamma(self.inner_idx[:T]) * x, dim=1)

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        B, T, F, D = x.shape
        z = self.compute_z(x)
        memory = self.gamma(self.outer_idx[:T]) * z + memory * self.gamma(
            self.state_offset[:T]
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x + memory * self.gamma(self.one, clamp=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        assert memory.dtype in [
            torch.complex64,
            torch.complex128,
        ], "State should be complex dtype"
        assert x.dim() == 3
        assert memory.dim() == 4

        B, T, F = x.shape
        x = x.reshape(B, T, F, 1)

        if x.shape[1] == 1:
            # More efficient shortcut for single-timestep inference
            return self.single_step_update(x, memory)
        elif x.shape[1] < self.max_len:
            # Default case, the whole thing can fit into a single temporal batch
            return self.batched_recurrent_update(x, memory)
        else:
            # Need to break into temporal batches
            chunks = x.split(self.max_len, dim=1)
            states = []
            for chunk in chunks:
                memory = self.batched_recurrent_update(chunk, memory[:, -1:])
                states.append(memory)
            return torch.cat(states, dim=1)


class LogspaceFFA(FFA):
    """FFA but designed for very long sequences using logspace arithmetic"""

    def set_nonzero(self, x, eps=1e-10):
        x.real[x.real == 0] = eps
        mask = x.real.abs() < eps
        x.real[mask] = x.real[mask].sign() * eps
        return x

    def compute_z(self, x):
        B, T, F, D = x.shape
        log_divisor = self.log_gamma(self.inner_idx[T - 1 : T])
        log_z = log_divisor + torch.log(
            torch.cumsum(
                (x + self.log_gamma(self.inner_idx[:T]) - log_divisor).exp(),
                dim=1,
            ),
        )
        return log_z

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        B, T, F, D = x.shape
        log_z = self.compute_z(x)
        memory = torch.exp(
            self.log_gamma(self.outer_idx[:T], clamp=False) + log_z
        ) + torch.exp(
            self.set_nonzero(memory).log()
            + self.log_gamma(self.state_offset[:T], clamp=False)
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return x.exp() + torch.exp(
            self.set_nonzero(memory).log() + self.log_gamma(self.one, clamp=False)
        )


# ────────────────────────────────────────────────────────────────────────────
# standalone/ffm/ffm.py (verbatim; relative import resolved by concatenation)
# ────────────────────────────────────────────────────────────────────────────
class FFM(nn.Module):
    """Fast and Forgetful Memory"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        context_size: int,
        output_size: int,
        min_period: int = 1,
        max_period: int = 1024,
        logspace: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size

        self.pre = nn.Linear(input_size, 2 * memory_size + 2 * output_size)
        if logspace:
            self.ffa = LogspaceFFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        else:
            self.ffa = FFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        self.mix = nn.Linear(2 * memory_size * context_size, output_size)
        self.ln_out = nn.LayerNorm(output_size, elementwise_affine=False)

    def initial_state(self, batch_size=1, device='cpu'):
        return torch.zeros(
            (batch_size, 1, self.memory_size, self.context_size),
            device=device,
            dtype=torch.complex64,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if state is None:
            s = self.initial_state(B, x.device)
        else:
            s = state

        # Compute values from x
        y, thru, gate = self.pre(x).split(
            [self.memory_size, self.output_size, self.memory_size + self.output_size],
            dim=-1,
        )

        # Compute gates
        gate = gate.sigmoid()
        in_gate, out_gate = gate.split([self.memory_size, self.output_size], dim=-1)

        # Compute state and output
        s = self.ffa((y * in_gate), s)  # Last dim for context
        z = self.mix(torch.view_as_real(s).reshape(B, T, -1))
        out = self.ln_out(z) * out_gate + thru * (1 - out_gate)

        return out, s


class DropInFFM(FFM):
    """Fast and Forgetful Memory, wrapped to behave like an nn.GRU"""

    def __init__(self, *args, batch_first=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if x missing singleton time or batch dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_y = True
        else:
            squeeze_y = False

        if self.batch_first:
            B, T, F = x.shape
        else:
            T, B, F = x.shape

        if state is None:
            s = self.initial_state(B, x.device)
        else:
            s = state

        s = s.reshape(B, 1, self.memory_size, self.context_size)

        assert s.shape == (
            B,
            1,
            self.memory_size,
            self.context_size,
        ), (
            f"Given x of shape {x.shape}, expected state to be"
            f" shape [{B}, 1, {self.memory_size}, {self.context_size}], dtype=complex, but got {s.shape} "
            f"and dtype={s.dtype}"
        )

        # Make everything batch first for FFM
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        # Call FFM
        y, s = super().forward(x, s)

        if not self.batch_first:
            y = y.permute(1, 0, 2)

        if squeeze_y:
            y = y.reshape(B, -1)

        # Return only terminal state of s
        s = s[:, -1]

        return y, s
