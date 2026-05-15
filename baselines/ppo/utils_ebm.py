"""
Shared utilities for EBM-Robo + dual-camera baselines.

Multi-camera observation wrapper. Drop-in replacement for the per-baseline
FlattenRGBDObservationWrapper (defined inside ppo_memtasks_*.py), but accepting
an ordered list of camera names. Cameras are concatenated along the channel
axis, in the order listed. All other return keys (state, joints, oracle_info,
prompt) are byte-identical to the single-camera version.

Usage:
    from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
    envs = FlattenRGBDObservationWrapperMulti(
        envs, rgb=True, depth=False, state=True, oracle=False, joints=True,
        target_cameras=["base_camera", "hand_camera"],
    )

The output `rgb` key has shape (..., H, W, 3 * num_cameras), in camera order.
Models can split with `rgb[..., 3*i:3*(i+1)]` to recover per-camera tensors.
"""

from __future__ import annotations

from typing import Dict, Sequence

import gymnasium as gym
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper


class FlattenRGBDObservationWrapperMulti(gym.ObservationWrapper):
    """Multi-camera version of FlattenRGBDObservationWrapper.

    Args:
        target_cameras: ordered list of camera names. RGB (and depth, if
            requested) tensors from each camera are concatenated channel-wise.
    """

    def __init__(
        self,
        env,
        rgb: bool = True,
        depth: bool = True,
        state: bool = True,
        oracle: bool = False,
        joints: bool = False,
        target_cameras: Sequence[str] = ("base_camera", "hand_camera"),
    ) -> None:
        self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints
        self.target_cameras = list(target_cameras)
        if not self.target_cameras:
            raise ValueError("target_cameras must contain at least one camera name")

        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        ret = dict()

        if self.include_rgb or self.include_depth:
            # oracle_info/prompt may be absent in raw ManiSkill obs; tolerate.
            if "oracle_info" in observation:
                ret["oracle_info"] = observation["oracle_info"]
            if "prompt" in observation:
                ret["prompt"] = observation["prompt"]
            sensor_data = observation.pop("sensor_data")
            if "sensor_param" in observation:
                del observation["sensor_param"]

            rgb_imgs, depth_imgs = [], []
            for cam_name in self.target_cameras:
                cam_data = sensor_data.get(cam_name)
                if cam_data is None:
                    raise KeyError(
                        f"target camera '{cam_name}' not found in sensor_data; "
                        f"available: {list(sensor_data.keys())}"
                    )
                if self.include_rgb and "rgb" in cam_data:
                    rgb_img = cam_data["rgb"]
                    if isinstance(rgb_img, torch.Tensor) and rgb_img.shape[-1] > 3:
                        rgb_img = rgb_img[..., :3]
                    rgb_imgs.append(rgb_img)
                if self.include_depth and "depth" in cam_data:
                    depth_imgs.append(cam_data["depth"])

            images = []
            if self.include_rgb and rgb_imgs:
                images.append(torch.concat(rgb_imgs, axis=-1))
            if self.include_depth and depth_imgs:
                images.append(torch.concat(depth_imgs, axis=-1))
            if images:
                images = torch.concat(images, axis=-1)
            else:
                images = None

        # Identical post-processing of state/joints/oracle as the single-cam wrapper.
        if self.include_state and not (self.include_rgb or self.include_depth):
            if not self.include_oracle:
                observation.pop("oracle_info")
        else:
            if not self.include_joints:
                filtered_obs = {k: v for k, v in observation.items() if k not in ["prompt", "oracle_info"]}
            else:
                extra_agent = {}
                for key in ["extra", "agent"]:
                    if key in observation:
                        extra_agent[key] = observation.pop(key)
                extra_agent_flat = common.flatten_state_dict(
                    extra_agent, use_torch=True, device=self.base_env.device
                )
                ret["joints"] = extra_agent_flat
                filtered_obs = {k: v for k, v in observation.items()
                                if k not in ["prompt", "oracle_info", "extra"]}

            observation = common.flatten_state_dict(
                filtered_obs, use_torch=True, device=self.base_env.device
            )

        if self.include_state and not (self.include_rgb or self.include_depth):
            ret = observation
        else:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images

        if "state" in ret.keys() and not self.include_state:
            ret.pop("state")
        if "oracle_info" in ret.keys() and not self.include_oracle and ret["oracle_info"] is not None:
            ret.pop("oracle_info")
        if "oracle_info" in ret.keys() and (ret["oracle_info"] == 4242424242).any().item():
            ret.pop("oracle_info")
        if "prompt" in ret.keys() and (ret["prompt"] == 4242424242).any().item():
            ret.pop("prompt")
        if "joints" in ret.keys() and not self.include_joints:
            ret.pop("joints")

        return ret

    @property
    def num_cameras(self) -> int:
        return len(self.target_cameras)

    def split_rgb_per_camera(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """Split the channel-concatenated rgb tensor back into per-camera tensors.

        rgb: (..., H, W, 3*N) -> list of N tensors each (..., H, W, 3).
        """
        n = self.num_cameras
        return [rgb[..., 3 * i : 3 * (i + 1)] for i in range(n)]
