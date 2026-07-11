"""Roll out IMed eval episodes with a trained V3a-LSTM-Hybrid agent and cache
the frames + ball positions, in the same .npz schema as path_a_data/RC9.

base_rgb, hand_rgb: (T, 128, 128, 3)
color_idx:         () int — set to 0 (the ball is red, MIKASA_COLORS[0] = (255,0,0))
                              so the same color_mask()-based ground truth works.
step_idx:          (T,) int
ep_seed:           ()  int
ball_xyz:          (T, 3) — env state ground truth
tcp_xyz:           (T, 3) — gripper position
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs import *  # noqa
from mikasa_robo_suite.utils.wrappers import (
    InitialZeroActionWrapper, RenderStepInfoWrapper,
    RenderRewardInfoWrapper, DebugRewardWrapper, StateOnlyTensorToDictWrapper,
)
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
from baselines.ppo.modules import EBMHybridLSTMMemoryModule

DEVICE = torch.device("cuda")

ENV_ID = "InterceptMedium-v0"
CKPT = ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptMedium-v0/ppo-v3a-lstm-imed-seed42__42__rgb_joints__20260517_145029/20260517_145029/final_ckpt.pt"
OUT_DIR = ROOT / "analysis/ebm/path_a_data" / ENV_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ENVS = 16
NUM_EPISODES = 20    # target across all envs


def build_env():
    env = gym.make(ENV_ID, num_envs=NUM_ENVS, reconfiguration_freq=1,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   render_mode="all", sim_backend="gpu", reward_mode="normalized_dense")
    for W, kw in [(StateOnlyTensorToDictWrapper, {}),
                  (InitialZeroActionWrapper, {"n_initial_steps": 0}),
                  (RenderStepInfoWrapper, {}), (RenderRewardInfoWrapper, {}),
                  (DebugRewardWrapper, {})]:
        env = W(env, **kw)
    env = FlattenRGBDObservationWrapperMulti(
        env, rgb=True, depth=False, state=False, oracle=False, joints=True,
        target_cameras=("base_camera", "hand_camera"),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, NUM_ENVS, ignore_terminations=True, record_metrics=True)


def make_agent(env, sample_obs):
    proprio_dim = sample_obs["joints"].shape[-1]
    ebm = EBMHybridLSTMMemoryModule(
        num_envs=NUM_ENVS, proprio_dim=proprio_dim,
        saliency_ckpt=str(ROOT / "analysis/ebm/path_a_head_v3.pt"),
        vit_backbone="dinov2", L=64, K=8, d_state=256,
        novelty_thresh=0.95, tau_age=30.0, device="cuda", no_saliency=False)

    class A(nn.Module):
        def __init__(s):
            super().__init__(); s.ebm = ebm
            from baselines.ppo.ppo_memtasks_ebm import layer_init
            act_dim = int(np.prod(env.unwrapped.single_action_space.shape))
            s.critic = nn.Sequential(layer_init(nn.Linear(256, 512)), nn.ReLU(),
                                     layer_init(nn.Linear(512, 1)))
            s.actor_mean = nn.Sequential(layer_init(nn.Linear(256, 512)), nn.ReLU(),
                                          layer_init(nn.Linear(512, act_dim),
                                                     std=0.01 * np.sqrt(2)), nn.Tanh())
            s.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
    agent = A().to(DEVICE)
    sd = torch.load(str(CKPT), map_location=DEVICE)
    own = {k: v.shape for k, v in agent.state_dict().items()}
    filt = {k: v for k, v in sd.items() if k not in own or v.shape == own[k]}
    agent.load_state_dict(filt, strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def collect():
    env = build_env()
    obs, _ = env.reset(seed=12345)
    agent = make_agent(env, obs)
    agent.ebm.reset(torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE))

    inner = env._env.unwrapped
    pending = [{"base": [], "hand": [], "ball": [], "tcp": [], "step": []} for _ in range(NUM_ENVS)]
    saved = 0
    done_prev = torch.zeros(NUM_ENVS, device=DEVICE)
    t_global = 0
    while saved < NUM_EPISODES:
        agent.ebm.reset(done_prev.bool())
        s_t, _ = agent.ebm.step(obs["rgb"], obs["joints"], t=t_global)
        ball_p = inner.ball.pose.p.cpu().numpy()
        tcp_p = inner.agent.tcp.pose.p.cpu().numpy()
        # split the 6-channel rgb back into base + hand for saving (each is 3 ch)
        rgb6 = obs["rgb"].cpu().numpy()  # (B, 128, 128, 6)
        base = rgb6[..., :3].astype(np.uint8)
        hand = rgb6[..., 3:].astype(np.uint8)
        for e in range(NUM_ENVS):
            pending[e]["base"].append(base[e])
            pending[e]["hand"].append(hand[e])
            pending[e]["ball"].append(ball_p[e])
            pending[e]["tcp"].append(tcp_p[e])
            pending[e]["step"].append(t_global)
        act = agent.actor_mean(s_t)
        obs, _, term, trunc, infos = env.step(act)
        done = torch.logical_or(term, trunc)
        if "final_info" in infos:
            done_mask = infos["_final_info"]
            for e in range(NUM_ENVS):
                if not bool(done_mask[e].item()): continue
                if saved >= NUM_EPISODES: break
                arr = pending[e]
                np.savez(OUT_DIR / f"ep{saved:03d}.npz",
                         base_rgb=np.stack(arr["base"]),
                         hand_rgb=np.stack(arr["hand"]),
                         color_idx=np.int32(0),      # ball is red
                         step_idx=np.array(arr["step"], dtype=np.int32),
                         ep_seed=np.int32(saved),
                         ball_xyz=np.stack(arr["ball"]),
                         tcp_xyz=np.stack(arr["tcp"]))
                print(f"  ep{saved:03d}: T={len(arr['base'])}")
                saved += 1
                pending[e] = {"base": [], "hand": [], "ball": [], "tcp": [], "step": []}
        done_prev = done.float()
        t_global = (t_global + 1) % 1000

    print(f"[cache] saved {saved} episodes to {OUT_DIR}")


if __name__ == "__main__":
    collect()
