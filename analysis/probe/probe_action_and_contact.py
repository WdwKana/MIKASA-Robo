"""
Action-smoothness + contact-state probe.

Diagnoses WHY V3a-LSTM has higher SR_end than V1 / V2-GRU-Hyb on IMed even
though all three methods encode ball position/velocity equally well
(see ball-state probe). The hypothesis under test:

  - V3a's policy produces smoother actions, especially when close to the
    ball, leading to cleaner interception. ‖a_t - a_{t-1}‖² should be
    lower for V3a, particularly during "near-contact" steps.
  - V3a's s_t encodes the gripper↔ball relation (tcp_to_ball) better than
    V1, even though ball-pose probing is equal across methods. That would
    point to "agent-relative contact awareness" as the missing ingredient.

Outputs JSON per checkpoint: action smoothness summary + tcp_to_ball
probe R² (linear and MLP).

Run on V1 / V2 / V3a × 3 seeds → 9 probes.
"""
from __future__ import annotations

import argparse
import json
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
    for W, kw in [
        (StateOnlyTensorToDictWrapper, {}),
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
        (DebugRewardWrapper, {}),
    ]:
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


def get_env_state(env) -> dict:
    inner = env._env.unwrapped
    return {
        "ball_pos":  inner.ball.pose.p,                       # (B, 3)
        "tcp_pos":   inner.agent.tcp.pose.p,                  # (B, 3)
        "ball_vel":  inner.ball.linear_velocity,              # (B, 3)
    }


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
    own_shapes = {k: v.shape for k, v in agent.state_dict().items()}
    filtered = {k: v for k, v in sd.items()
                if k not in own_shapes or v.shape == own_shapes[k]}
    agent.load_state_dict(filtered, strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def collect_rollouts(agent, env, num_episodes: int, num_envs: int,
                     max_steps_per_episode: int = 90):
    """Returns dict with arrays over (episode, step):
      - s:           (N_steps, D)
      - a:           (N_steps, A)
      - tcp_to_ball: (N_steps, 3)
      - dist:        (N_steps,)
      - step:        (N_steps,)  step index in episode (0..max_steps-1)
      - episode_id:  (N_steps,)
      - success_end_per_step: (N_steps,) — broadcasted episode-final success_at_end flag
    """
    obs, _ = env.reset(seed=12345)
    agent.ebm.reset(torch.ones(num_envs, dtype=torch.bool, device=DEVICE))

    s_list, a_list, tcp2ball_list = [], [], []
    step_list, ep_list = [], []
    success_list = []
    in_contact_first_step = torch.full((num_envs,), -1, device=DEVICE, dtype=torch.long)
    success_end_per_env = []  # filled at episode end

    episode_id_counter = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    step_in_ep = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    done_prev = torch.zeros(num_envs, device=DEVICE)
    episodes_done = 0
    t_global = 0

    # Buffer per-step records; on episode end we tag each step with success.
    pending_records = [[] for _ in range(num_envs)]  # list per env of (s, a, tcp2ball)

    while episodes_done < num_episodes:
        agent.ebm.reset(done_prev.bool())
        s_t, _ = agent.ebm.step(obs["rgb"], obs["joints"], t=t_global)
        st = get_env_state(env)
        tcp2ball = st["ball_pos"] - st["tcp_pos"]  # (B, 3)

        act = agent.actor_mean(s_t)
        # record BEFORE stepping (s_t, a_t, env state at decision time)
        s_np = s_t.detach().cpu().numpy()
        a_np = act.detach().cpu().numpy()
        t2b_np = tcp2ball.detach().cpu().numpy()
        step_np = step_in_ep.cpu().numpy()
        epid_np = episode_id_counter.cpu().numpy()

        for e in range(num_envs):
            pending_records[e].append(
                (s_np[e], a_np[e], t2b_np[e], int(step_np[e]), int(epid_np[e]))
            )

        obs, rew, term, trunc, infos = env.step(act)
        done = torch.logical_or(term, trunc)
        # On episode end, harvest success_at_end from final_info
        if "final_info" in infos:
            final = infos["final_info"]
            done_mask = infos["_final_info"]
            # success_at_end is a per-env flag for envs that just ended
            sr_end = final["episode"].get("success_at_end")
            if sr_end is not None:
                for e in range(num_envs):
                    if not bool(done_mask[e].item()):
                        continue
                    is_succ = float(sr_end[e].item())
                    # tag all pending records for this env as that success_end
                    for rec in pending_records[e]:
                        s_list.append(rec[0])
                        a_list.append(rec[1])
                        tcp2ball_list.append(rec[2])
                        step_list.append(rec[3])
                        ep_list.append(rec[4])
                        success_list.append(is_succ)
                    pending_records[e] = []
                    episode_id_counter[e] += 1
                    episodes_done += 1

        done_prev = done.float()
        step_in_ep = torch.where(done, torch.zeros_like(step_in_ep),
                                 step_in_ep + 1)
        t_global = (t_global + 1) % 1000

    S = np.array(s_list)
    A = np.array(a_list)
    T2B = np.array(tcp2ball_list)
    STEP = np.array(step_list)
    EP = np.array(ep_list)
    SUCC = np.array(success_list)
    return {
        "s": S, "a": A, "tcp_to_ball": T2B,
        "step": STEP, "episode_id": EP, "success_end": SUCC,
    }


def fit_probe_linear(X, Y, ridge=1e-2):
    N = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N); s = int(0.8*N)
    tr, te = perm[:s], perm[s:]
    Xtr = np.concatenate([X[tr], np.ones((s, 1))], axis=1)
    Xte = np.concatenate([X[te], np.ones((N-s, 1))], axis=1)
    D = Xtr.shape[1]
    A = Xtr.T @ Xtr + ridge*np.eye(D); A[-1,-1] -= ridge
    W = np.linalg.solve(A, Xtr.T @ Y[tr])
    Yhat = Xte @ W; Yte = Y[te]
    ss_res = ((Yte-Yhat)**2).sum(axis=0); ss_tot = ((Yte-Yte.mean(0))**2).sum(axis=0)
    r2_per = 1 - ss_res / np.maximum(ss_tot, 1e-8)
    return {
        "r2_per_axis": r2_per.tolist() if Y.ndim==2 else [float(r2_per)],
        "r2_overall": float(1 - ss_res.sum() / max(ss_tot.sum(),1e-8)),
    }


def fit_probe_mlp(X, Y, hidden=256, epochs=80, lr=1e-3, batch=1024):
    N, D = X.shape
    if Y.ndim == 1: Y = Y[:, None]
    K = Y.shape[1]
    rng = np.random.default_rng(0); perm = rng.permutation(N); s = int(0.8*N)
    tr, te = perm[:s], perm[s:]
    Xtr = torch.from_numpy(X[tr]).float().to(DEVICE)
    Ytr = torch.from_numpy(Y[tr]).float().to(DEVICE)
    Xte = torch.from_numpy(X[te]).float().to(DEVICE)
    Yte = torch.from_numpy(Y[te]).float().to(DEVICE)
    m = nn.Sequential(
        nn.Linear(D, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, K),
    ).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    n_tr = Xtr.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, batch):
            b = idx[i:i+batch]
            loss = ((m(Xtr[b]) - Ytr[b])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): Yhat = m(Xte).cpu().numpy()
    Yte_np = Yte.cpu().numpy()
    ss_res = ((Yte_np-Yhat)**2).sum(axis=0); ss_tot = ((Yte_np-Yte_np.mean(0))**2).sum(axis=0)
    r2_per = 1 - ss_res / np.maximum(ss_tot, 1e-8)
    return {
        "r2_per_axis": r2_per.tolist(),
        "r2_overall": float(1 - ss_res.sum() / max(ss_tot.sum(),1e-8)),
    }


def action_smoothness(data: dict, near_threshold: float = 0.10):
    """Compute mean ‖a_t - a_{t-1}‖² overall and conditional on:
      - near-ball (|tcp_to_ball| < near_threshold)
      - far-from-ball
      - success episode vs failure episode
    """
    a = data["a"]
    step = data["step"]
    ep = data["episode_id"]
    dist = np.linalg.norm(data["tcp_to_ball"], axis=1)
    succ = data["success_end"]

    # Compute per-step ‖a_t - a_{t-1}‖² (skip step 0 within each episode)
    deltas = np.full(a.shape[0], np.nan)
    for i in range(1, a.shape[0]):
        if ep[i] != ep[i-1]:
            continue
        deltas[i] = np.sum((a[i] - a[i-1])**2)
    valid = ~np.isnan(deltas)

    def stat(mask):
        d = deltas[mask & valid]
        if d.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {"mean": float(d.mean()), "median": float(np.median(d)), "n": int(d.size)}

    near = dist < near_threshold
    succ_mask = succ > 0.5
    return {
        "overall":         stat(np.ones_like(valid, dtype=bool)),
        "near_ball":       stat(near),
        "far_from_ball":   stat(~near),
        "success_eps":     stat(succ_mask),
        "failure_eps":     stat(~succ_mask),
        "near_in_success": stat(near & succ_mask),
        "near_in_failure": stat(near & ~succ_mask),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="InterceptMedium-v0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episodes", type=int, default=32)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print(f"[probe2] method={args.method} ckpt={args.checkpoint}")
    env = build_env(args.env_id, args.num_envs)
    obs, _ = env.reset(seed=0)
    agent = make_agent(args.method, args.checkpoint, env, obs, args.num_envs)

    print(f"[probe2] collecting {args.num_episodes} episodes...")
    data = collect_rollouts(agent, env, args.num_episodes, args.num_envs)
    print(f"[probe2] N={data['s'].shape[0]} steps, s_t={data['s'].shape[1]}d, "
          f"a={data['a'].shape[1]}d, success_rate_in_data="
          f"{(data['success_end']>0.5).mean():.3f}")

    # Action smoothness
    sm = action_smoothness(data, near_threshold=0.10)

    # Probe s_t -> tcp_to_ball (vector) and -> distance
    dist = np.linalg.norm(data["tcp_to_ball"], axis=1, keepdims=True).astype(np.float32)
    t2b_vec_lin = fit_probe_linear(data["s"], data["tcp_to_ball"])
    t2b_vec_mlp = fit_probe_mlp(data["s"], data["tcp_to_ball"])
    dist_lin = fit_probe_linear(data["s"], dist)
    dist_mlp = fit_probe_mlp(data["s"], dist)

    out = {
        "method":      args.method,
        "checkpoint":  args.checkpoint,
        "env_id":      args.env_id,
        "num_steps":   int(data["s"].shape[0]),
        "num_episodes": int(data["episode_id"].max()) + 1,
        "success_rate_in_data": float((data["success_end"] > 0.5).mean()),
        "action_smoothness":   sm,
        "tcp_to_ball_vec_linear":  t2b_vec_lin,
        "tcp_to_ball_vec_mlp":     t2b_vec_mlp,
        "tcp_to_ball_dist_linear": dist_lin,
        "tcp_to_ball_dist_mlp":    dist_mlp,
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("checkpoint",)}, indent=2, default=float))
    out_path = args.output or str(Path(args.checkpoint).parent / "probe2_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[probe2] saved to {out_path}")


if __name__ == "__main__":
    main()
