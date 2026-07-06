"""VENDORED baseline code — Stable Hadamard Memory (SHM).

Source: https://github.com/thaihungle/SHM  (MIT License)
Paper:  "Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for
        Reinforcement Learning", Le et al., arXiv:2410.10132.

The memory-cell math below is copied VERBATIM from
`popgym/baselines/ray_models/ray_shm.py` (class SHMAgent) — the implementation
that produced the paper's POPGym results via `train_popgym.py` (the root-level
`shm.py` standalone demo differs: it lacks the `shortcut` residual read-out
present here). Only the Ray/rllib scaffolding (BaseModel, config plumbing) is
removed; layer definitions, `retrieve_theta`, and `matrix_memory_update` are
untouched, including two notable properties of the released code we deliberately
do NOT fix (baseline fidelity — any deviation would make results ours, not theirs):

  1. `retrieve_theta` draws `torch.empty(...).uniform_(0, 1).long()`, which is
     ALWAYS 0 — so the paper's per-step random selection over L=2^7 calibration
     rows degenerates to always using `theta_matrix[0]` (a single learned
     vector). We report the baseline with the authors' code as released.
  2. `torch.nn.init.xavier_uniform` (deprecated alias, no trailing underscore)
     is kept as-is.

Experiment defaults from the authors' `train_popgym.py` argparse: mem_size
(H in the paper) = 72, post_size = 1024. The authors' sine positional embedding
belongs to their rllib BaseModel preprocessing (not the memory cell) and is NOT
used here — none of our perception-matched baselines add positional embeddings.
Adaptation to our PPO harness lives in `recurrent_adapters.py`.
"""
# flake8: noqa
import torch
from torch import nn


class SHMCell(nn.Module):
    """The SHM memory cell (verbatim math from ray_shm.py::SHMAgent)."""

    def __init__(self, input_size: int, mem_size: int = 72, post_size: int = 1024,
                 add_space: int = 7):
        super().__init__()
        self.add_space = add_space
        self.L = 2 ** self.add_space
        self.mem_size = mem_size  # H in the paper
        self.key = nn.Linear(input_size, self.mem_size, bias=False)
        self.query = nn.Linear(input_size, self.mem_size, bias=False)
        self.value = nn.Linear(input_size, self.mem_size, bias=False)
        self.vc_net = nn.Linear(input_size, self.mem_size, bias=False)
        self.eta = nn.Linear(input_size, 1, bias=False)
        self.theta_matrix = nn.Parameter(torch.rand(self.L, self.mem_size))
        torch.nn.init.xavier_uniform(self.theta_matrix)
        self.shortcut = nn.Linear(input_size, self.mem_size)
        self.out = nn.Linear(mem_size, post_size)
        self.norm = nn.LayerNorm(input_size)

    def retrieve_theta(self, x, theta_matrix):
        B, T, D = x.shape
        a = torch.empty(x.shape[0], x.shape[1]).uniform_(0, 1).long().to(x.device)  # generate a uniform random matrix with range [0, 1]
        attention = torch.nn.functional.one_hot(a, num_classes=self.L).float()
        theta = torch.einsum("ij, bti -> btj", theta_matrix, attention)
        return theta

    def matrix_memory_update(self, x, state):
        x = self.norm(x)
        B, T, d = x.shape  # B, T, d

        K = self.key(x).relu()
        Q = self.query(x).relu()

        eta1 = torch.sigmoid(self.eta(x))
        vc = self.vc_net(x)
        theta = self.retrieve_theta(x, self.theta_matrix)
        C = torch.einsum("bti, btj -> btij", theta, vc)
        C = 1.0 + torch.tanh(C)

        K = K / (1e-5 + K.sum(dim=-1, keepdim=True))
        Q = Q / (1e-5 + Q.sum(dim=-1, keepdim=True))
        V = self.value(x)

        state = state.squeeze(1)
        states = []

        # Memory writting: naive sequential implementation
        for t in range(T):
            state = state * C[:, t] + torch.einsum("bi, bj -> bij", eta1[:, t] * V[:, t], K[:, t, :])
            states.append(state)
        state = torch.stack(states, dim=1)

        # Memory reading: following original FWP code of the benchmark
        y = torch.einsum("btij, btj -> bti", state, Q)
        y = y + self.shortcut(x)
        return y, state

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """x: [B, T, input_size]; state: [B, mem_size, mem_size].
        Returns y: [B, T, post_size]; state: [B, mem_size, mem_size] (terminal).
        Mirrors ray_shm.py::forward_memory (read via Q, add shortcut, project out)."""
        y, states = self.matrix_memory_update(x, state)
        z = self.out(y)
        return z, states[:, -1]
