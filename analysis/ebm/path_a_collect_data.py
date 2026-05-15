"""
Path A — Step 1: collect (base_rgb, hand_rgb, target_color_idx, step) per frame
                via RANDOM-policy rollout on RememberColor9-v0.

MIKASA target objects (color cubes here) are placed by the env regardless of
policy, so random actions give us diverse images of targets and candidates
without needing the oracle PPO checkpoint.

Output: analysis/ebm/path_a_data/RememberColor9-v0/{ep:03d}.npz
   per file:
     base_rgb : (T, 128, 128, 3) uint8
     hand_rgb : (T, 128, 128, 3) uint8
     color_idx: int  (0..8) per episode
     step_idx : (T,) int

20 episodes × ~60 steps ≈ 1200 frames. A few minutes on GPU.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

import mikasa_robo_suite  # registers envs
from mikasa_robo_suite.utils.wrappers import RememberColorInfoWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="RememberColor9-v0")
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.env_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing -> {out_dir}")

    env = gym.make(
        args.env_id, num_envs=1, obs_mode="rgb",
        render_mode=None, control_mode="pd_joint_delta_pos", sim_backend="gpu",
    )
    env = RememberColorInfoWrapper(env)

    spec = gym.spec(args.env_id)
    max_steps = int(spec.max_episode_steps)
    print(f"episode max steps = {max_steps}")

    rng = np.random.default_rng(args.seed)
    saved = 0

    while saved < args.num_episodes:
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=ep_seed)
        # oracle_info lives in info; for RememberColor it is the target color index
        target_idx = int(info["oracle_info"][0].item())

        base_rgbs, hand_rgbs, step_idx = [], [], []
        # capture step 0 (initial frame after wrapper resets)
        sd = obs["sensor_data"]
        base_rgbs.append(sd["base_camera"]["rgb"][0].cpu().numpy())
        hand_rgbs.append(sd["hand_camera"]["rgb"][0].cpu().numpy())
        step_idx.append(0)

        for t in range(1, max_steps):
            act = torch.from_numpy(env.action_space.sample()).to(env.unwrapped.device)
            obs, _, term, trunc, info = env.step(act)
            sd = obs["sensor_data"]
            base_rgbs.append(sd["base_camera"]["rgb"][0].cpu().numpy())
            hand_rgbs.append(sd["hand_camera"]["rgb"][0].cpu().numpy())
            step_idx.append(t)
            if bool(term[0]) or bool(trunc[0]):
                break

        np.savez_compressed(
            out_dir / f"ep{saved:03d}.npz",
            base_rgb=np.stack(base_rgbs).astype(np.uint8),
            hand_rgb=np.stack(hand_rgbs).astype(np.uint8),
            color_idx=np.int32(target_idx),
            step_idx=np.array(step_idx, dtype=np.int32),
            ep_seed=np.int64(ep_seed),
        )
        print(f"  ep{saved:03d}  T={len(step_idx)}  target_color={target_idx}  seed={ep_seed}")
        saved += 1

    env.close()
    print(f"saved {saved} episodes to {out_dir}")


if __name__ == "__main__":
    main()
