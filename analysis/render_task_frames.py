"""Render clean task-illustration frames (no debug overlays) for the paper.

Makes each env directly via gym.make (no training wrappers -> no text overlay),
steps with zero actions, and saves the composite render at several timesteps.
Matches the style of the existing paper/Figures/color9_*.png illustrations.
"""
import os

import gymnasium as gym
import numpy as np
import mikasa_robo_suite  # noqa: F401  (registers envs)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "Figures", "task_frames")
os.makedirs(OUT, exist_ok=True)

TASKS = [
    ("RememberShape9-v0", "shape9", (0, 5, 10, 15, 20)),
    ("ShellGameTouch-v0", "shellgame_touch", (0, 5, 10, 15, 20, 25)),
]

for env_id, prefix, keep in TASKS:
    # kwargs identical to the training entries (phase logic depends on them)
    env = gym.make(env_id, num_envs=1, obs_mode="rgb", render_mode="all",
                   control_mode="pd_joint_delta_pos", sim_backend="gpu",
                   reward_mode="normalized_dense", reconfiguration_freq=1)
    env.reset(seed=7)
    import imageio
    for t in range(max(keep) + 1):
        img = env.render()
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:            # (num_envs, H, W, 3)
            img = img[0]
        if t in keep:
            path = os.path.join(OUT, f"{prefix}_{t}.png")
            imageio.imwrite(path, img.astype(np.uint8))
            print("saved", path, img.shape)
        a = env.action_space.sample() * 0
        env.step(a)
    env.close()
print("DONE")
