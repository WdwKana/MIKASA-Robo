"""
Linear-probe analysis: does the agent's internal state s_t encode the ball's
position and velocity?

Hypothesis from RESEARCH_DIRECTION.md:
  - Pure K-V buffer methods (V1, MemVLA) store snapshots; integrated
    quantities like velocity should be hard to recover linearly.
  - Recurrent methods (V2-Hybrid GRU, V3a LSTM, V4-belief variants) integrate
    motion in their hidden state; linear probe should recover velocity.

Procedure:
  1. Load a trained agent's checkpoint on InterceptMedium-v0.
  2. Run N eval episodes, recording s_t (from the agent's forward) plus the
     ground-truth ball pose / velocity (queried directly from the underlying
     SAPIEN actor).
  3. Train ridge regression s_t -> ball_xyz  and  s_t -> ball_vel_xyz.
  4. Report R² and per-axis MSE on a held-out split.

Output written next to the checkpoint as `probe_results.json`.

Currently supports V1 (EBMMemoryModule). The "method" arg is structured so
new variants drop in via the AGENT_BUILDERS dict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs import *  # noqa: F401
from mikasa_robo_suite.utils.wrappers import (
    InitialZeroActionWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
    DebugRewardWrapper,
    StateOnlyTensorToDictWrapper,
)
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
from baselines.ppo.modules import (
    EBMMemoryModule, EBMHybridMemoryModule,
    EBMHybridLSTMMemoryModule, EBMHybridCleanMemoryModule,
    EBMBeliefGRUMemoryModule, EBMBeliefLSTMMemoryModule,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


METHOD_MODULES = {
    "v1":  EBMMemoryModule,
    "v2":  EBMHybridMemoryModule,
    "v3a": EBMHybridLSTMMemoryModule,
    "v3b": EBMHybridCleanMemoryModule,
    "v4a": EBMBeliefGRUMemoryModule,
    "v4b": EBMBeliefLSTMMemoryModule,
}


def build_env(env_id: str, num_envs: int):
    env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos",
                      render_mode="all", sim_backend="gpu",
                      reward_mode="normalized_dense")
    env = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)
    wrappers = [
        (StateOnlyTensorToDictWrapper, {}),
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
        (DebugRewardWrapper, {}),
    ]
    for W, kw in wrappers:
        env = W(env, **kw)
    env = FlattenRGBDObservationWrapperMulti(
        env, rgb=True, depth=False, state=False,
        oracle=False, joints=True,
        target_cameras=("base_camera", "hand_camera"),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True,
                             record_metrics=True)
    return env


def get_ball_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (ball_xyz, ball_vel_xyz) of shape (num_envs, 3) each."""
    inner = env._env.unwrapped
    pos = inner.ball.pose.p              # (num_envs, 3) on CUDA
    vel = inner.ball.linear_velocity     # (num_envs, 3) on CUDA
    return pos, vel


def make_agent(method: str, checkpoint: str, envs, sample_obs, num_envs: int):
    ModCls = METHOD_MODULES[method]
    proprio_dim = sample_obs["joints"].shape[-1]
    ebm = ModCls(
        num_envs=num_envs, proprio_dim=proprio_dim,
        saliency_ckpt=str(ROOT / "analysis/ebm/path_a_head_v3.pt"),
        vit_backbone="dinov2", L=64, K=8, d_state=256,
        novelty_thresh=0.95, tau_age=30.0,
        device="cuda", no_saliency=False,
    )

    # Build a stub Agent matching the saved checkpoint (just ebm + actor_mean).
    class StubAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.ebm = ebm
            d_state = 256
            act_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
            from baselines.ppo.ppo_memtasks_ebm import layer_init
            self.critic = nn.Sequential(
                layer_init(nn.Linear(d_state, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, 1)),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(d_state, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, act_dim), std=0.01 * np.sqrt(2)),
                nn.Tanh(),
            )
            self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    agent = StubAgent().to(DEVICE)
    sd = torch.load(checkpoint, map_location=DEVICE)
    # Drop num_envs-shaped buffers (gru_state, buffer.features, etc.) — they
    # depend on training-time num_envs and are re-initialized at probe time.
    own_shapes = {k: v.shape for k, v in agent.state_dict().items()}
    filtered = {k: v for k, v in sd.items()
                if k not in own_shapes or v.shape == own_shapes[k]}
    dropped = sorted(set(sd.keys()) - set(filtered.keys()))
    if dropped:
        print(f"[probe] dropped {len(dropped)} shape-mismatched buffers: "
              f"{dropped[:3]}{'...' if len(dropped) > 3 else ''}")
    agent.load_state_dict(filtered, strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def collect_rollouts(agent, env, num_episodes: int, num_envs: int,
                     max_steps: int = 90, has_recurrent: bool = False):
    """Returns (S_T: NxD, BALL: Nx3, VEL: Nx3) flattened across envs+timesteps."""
    obs, _ = env.reset(seed=12345)
    agent.ebm.reset(torch.ones(num_envs, dtype=torch.bool, device=DEVICE))

    s_list, ball_list, vel_list = [], [], []
    t_global = 0
    episodes_done = 0
    done_prev = torch.zeros(num_envs, device=DEVICE)

    while episodes_done < num_episodes:
        agent.ebm.reset(done_prev.bool())
        s_t, _ = agent.ebm.step(obs["rgb"], obs["joints"], t=t_global)
        ball_p, ball_v = get_ball_state(env)

        s_list.append(s_t.detach().cpu().numpy())
        ball_list.append(ball_p.detach().cpu().numpy())
        vel_list.append(ball_v.detach().cpu().numpy())

        act = agent.actor_mean(s_t)
        obs, _, term, trunc, _ = env.step(act)
        done_prev = torch.logical_or(term, trunc).float()
        t_global += 1
        if done_prev.any():
            episodes_done += int(done_prev.sum().item()) // num_envs
            t_global = 0  # rough; reset for the envs that just ended

    S = np.concatenate(s_list, axis=0)
    BALL = np.concatenate(ball_list, axis=0)
    VEL = np.concatenate(vel_list, axis=0)
    return S, BALL, VEL


def fit_probe(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-2):
    """Closed-form ridge regression with bias.

    X: (N, D)   features
    Y: (N, K)   targets
    returns R² per target dim and overall, plus MSE.
    """
    N = X.shape[0]
    # 80/20 split
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    s = int(0.8 * N)
    tr, te = perm[:s], perm[s:]
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]

    # add bias
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)

    # W = (X^T X + λI)^-1 X^T Y
    D = Xtr1.shape[1]
    A = Xtr1.T @ Xtr1 + ridge * np.eye(D)
    A[-1, -1] -= ridge  # don't regularize bias
    W = np.linalg.solve(A, Xtr1.T @ Ytr)

    Yhat = Xte1 @ W
    # R²
    ss_res = ((Yte - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    r2_per = 1 - ss_res / np.maximum(ss_tot, 1e-8)
    r2_overall = 1 - ss_res.sum() / max(ss_tot.sum(), 1e-8)
    mse = ((Yte - Yhat) ** 2).mean(axis=0)
    return {
        "r2_per_axis":  r2_per.tolist(),
        "r2_overall":   float(r2_overall),
        "mse_per_axis": mse.tolist(),
        "n_train":      int(s),
        "n_test":       int(N - s),
    }


def fit_probe_mlp(X: np.ndarray, Y: np.ndarray,
                  hidden: int = 256, epochs: int = 100, lr: float = 1e-3,
                  weight_decay: float = 1e-4, batch_size: int = 1024):
    """2-layer MLP probe. Catches non-linear structure that ridge misses.

    Reports R² and MSE on a held-out 20% split (same split as fit_probe).
    """
    N, D = X.shape
    K = Y.shape[1]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    s = int(0.8 * N)
    tr, te = perm[:s], perm[s:]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X[tr]).float().to(dev)
    Ytr = torch.from_numpy(Y[tr]).float().to(dev)
    Xte = torch.from_numpy(X[te]).float().to(dev)
    Yte = torch.from_numpy(Y[te]).float().to(dev)

    model = nn.Sequential(
        nn.Linear(D, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, K),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_tr = Xtr.shape[0]
    for ep in range(epochs):
        order = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, batch_size):
            idx = order[i:i+batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        Yhat = model(Xte).cpu().numpy()
    Yte_np = Yte.cpu().numpy()
    ss_res = ((Yte_np - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Yte_np - Yte_np.mean(axis=0)) ** 2).sum(axis=0)
    r2_per = 1 - ss_res / np.maximum(ss_tot, 1e-8)
    r2_overall = 1 - ss_res.sum() / max(ss_tot.sum(), 1e-8)
    mse = ((Yte_np - Yhat) ** 2).mean(axis=0)
    return {
        "r2_per_axis":  r2_per.tolist(),
        "r2_overall":   float(r2_overall),
        "mse_per_axis": mse.tolist(),
        "hidden":       int(hidden),
        "epochs":       int(epochs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHOD_MODULES.keys()))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="InterceptMedium-v0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episodes", type=int, default=64)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print(f"[probe] method={args.method} ckpt={args.checkpoint}")
    env = build_env(args.env_id, args.num_envs)
    obs, _ = env.reset(seed=0)
    agent = make_agent(args.method, args.checkpoint, env, obs, args.num_envs)

    # has_recurrent is informational only; collect_rollouts handles both.
    has_rec = args.method != "v1"

    print(f"[probe] collecting {args.num_episodes} episodes...")
    S, BALL, VEL = collect_rollouts(
        agent, env, num_episodes=args.num_episodes, num_envs=args.num_envs,
        has_recurrent=has_rec,
    )
    print(f"[probe] collected {S.shape[0]} (state, ball) pairs; s_t dim = {S.shape[1]}")

    print("[probe] fitting linear (ridge) probes...")
    pos_results = fit_probe(S, BALL)
    vel_results = fit_probe(S, VEL)
    print("[probe] fitting MLP probes (non-linear)...")
    pos_mlp = fit_probe_mlp(S, BALL)
    vel_mlp = fit_probe_mlp(S, VEL)
    result = {
        "method":      args.method,
        "checkpoint":  args.checkpoint,
        "env_id":      args.env_id,
        "num_samples": int(S.shape[0]),
        "s_t_dim":     int(S.shape[1]),
        "ball_position":      pos_results,
        "ball_velocity":      vel_results,
        "ball_position_mlp":  pos_mlp,
        "ball_velocity_mlp":  vel_mlp,
    }
    print(json.dumps(result, indent=2))

    out_path = args.output or str(Path(args.checkpoint).parent / "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] saved to {out_path}")


if __name__ == "__main__":
    main()
