"""Generic eval-frame collector. Roll out a trained V1 checkpoint on a given
task and cache (base_rgb, hand_rgb, color_idx, step_idx, ep_seed) per ep."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

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
from mikasa_robo_suite.utils.wrappers import (
    RememberColorInfoWrapper, RememberShapeInfoWrapper,
)
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
from baselines.ppo.modules import EBMMemoryModule

DEVICE = torch.device("cuda")
NUM_ENVS = 16


def build_env(env_id, info_wrapper):
    env = gym.make(env_id, num_envs=NUM_ENVS, reconfiguration_freq=1,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   render_mode="all", sim_backend="gpu", reward_mode="normalized_dense")
    for W, kw in [(StateOnlyTensorToDictWrapper, {}),
                  (InitialZeroActionWrapper, {"n_initial_steps": 0})]:
        env = W(env, **kw)
    env = info_wrapper(env)   # task-specific: color_idx / shape_idx info wrapper
    for W, kw in [(RenderStepInfoWrapper, {}), (RenderRewardInfoWrapper, {}),
                  (DebugRewardWrapper, {})]:
        env = W(env, **kw)
    env = FlattenRGBDObservationWrapperMulti(
        env, rgb=True, depth=False, state=False, oracle=False, joints=True,
        target_cameras=("base_camera", "hand_camera"),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, NUM_ENVS, ignore_terminations=True, record_metrics=True)


def make_agent(env, sample_obs, ckpt):
    proprio_dim = sample_obs["joints"].shape[-1]
    ebm = EBMMemoryModule(
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
    sd = torch.load(ckpt, map_location=DEVICE)
    own = {k: v.shape for k, v in agent.state_dict().items()}
    filt = {k: v for k, v in sd.items() if k not in own or v.shape == own[k]}
    agent.load_state_dict(filt, strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def collect(env_id, ckpt, num_eps, info_wrapper, color_info_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(env_id, info_wrapper)
    obs, _ = env.reset(seed=12345)
    agent = make_agent(env, obs, ckpt)
    agent.ebm.reset(torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE))
    inner = env._env.unwrapped
    pending = [{"base": [], "hand": [], "step": []} for _ in range(NUM_ENVS)]
    saved = 0
    done_prev = torch.zeros(NUM_ENVS, device=DEVICE)
    t_global = 0
    while saved < num_eps:
        agent.ebm.reset(done_prev.bool())
        s_t, _ = agent.ebm.step(obs["rgb"], obs["joints"], t=t_global)
        rgb6 = obs["rgb"].cpu().numpy()
        base = rgb6[..., :3].astype(np.uint8)
        hand = rgb6[..., 3:].astype(np.uint8)
        for e in range(NUM_ENVS):
            pending[e]["base"].append(base[e])
            pending[e]["hand"].append(hand[e])
            pending[e]["step"].append(t_global)
        act = agent.actor_mean(s_t)
        obs, _, term, trunc, infos = env.step(act)
        done = torch.logical_or(term, trunc)
        if "final_info" in infos:
            done_mask = infos["_final_info"]
            # color_info_key in final_info or initial info — use the wrapper-attached field
            for e in range(NUM_ENVS):
                if not bool(done_mask[e].item()): continue
                if saved >= num_eps: break
                # try to read color/shape index from obs prompt (set by info wrapper)
                # the info wrapper appends to "prompt" or "oracle_info"; for simplicity
                # we use inner state directly
                if hasattr(inner, "true_color_indices"):
                    color = int(inner.true_color_indices[e].item())
                elif hasattr(inner, "true_shape_indices"):
                    color = int(inner.true_shape_indices[e].item())
                else:
                    color = 0
                arr = pending[e]
                np.savez(out_dir / f"ep{saved:03d}.npz",
                         base_rgb=np.stack(arr["base"]),
                         hand_rgb=np.stack(arr["hand"]),
                         color_idx=np.int32(color),
                         step_idx=np.array(arr["step"], dtype=np.int32),
                         ep_seed=np.int32(saved))
                print(f"  ep{saved:03d}: T={len(arr['base'])} color={color}")
                saved += 1
                pending[e] = {"base": [], "hand": [], "step": []}
        done_prev = done.float()
        t_global = (t_global + 1) % 1000
    print(f"[cache] saved {saved} to {out_dir}")


if __name__ == "__main__":
    TASKS = {
        "rc5": ("RememberColor5-v0", RememberColorInfoWrapper,
                ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor5-v0/ppo-ebm-rc5-seed42__42__rgb_joints__20260512_181658/20260512_181658/final_ckpt.pt"),
        "shape5": ("RememberShape5-v0", RememberShapeInfoWrapper,
                   ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShape5-v0/ppo-ebm-shape5-seed42__42__rgb_joints__20260507_204219/20260507_204219/final_ckpt.pt"),
    }
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS.keys()), required=True)
    p.add_argument("--num-eps", type=int, default=20)
    args = p.parse_args()
    env_id, wrapper, ckpt = TASKS[args.task]
    out_dir = ROOT / "analysis/ebm/path_a_data" / env_id
    collect(env_id, str(ckpt), args.num_eps, wrapper, "color_idx", out_dir)
