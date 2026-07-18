from collections import defaultdict
import os
import random
import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

if os.path.exists("wandb_config.yaml"):
    import yaml
    with open("wandb_config.yaml") as f:
        wandb_config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Memory-ManiSkill specific imports
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *

# Dual-camera observation wrapper (base_camera + hand_camera)
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
# V2 (hybrid): use EBMHybridMemoryModule, aliased so the rest of this file's
# code (built from V1) refers to it as EBMMemoryModule with no other changes.
# Self-Referential Buffer: no saliency head, no predictor.
from baselines.ppo.modules.ebm_srb_tr_nolstm_cres import EBMSRBTRNoLSTMCRESMemoryModule as EBMMemoryModule


import copy
from typing import Dict
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='.*env\\.\\w+ to get variables from other wrappers is deprecated.*')

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env, rgb=True, depth=True, state=True, oracle=False, joints=False, target_camera="base_camera") -> None:
        self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints
        self.target_camera = target_camera

        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        # Save oracle_info if it exists
        ret = dict()

        if self.include_rgb or self.include_depth:
            ret['oracle_info'] = observation['oracle_info']
            ret['prompt'] = observation['prompt']
            sensor_data = observation.pop("sensor_data")

            del observation["sensor_param"]
            images = []

            cam_data = sensor_data.get(self.target_camera)
            if cam_data is None and len(sensor_data) > 0:
                cam_data = next(iter(sensor_data.values()))

            if cam_data is not None:
                if self.include_rgb and "rgb" in cam_data:
                    rgb_img = cam_data["rgb"]
                    if isinstance(rgb_img, torch.Tensor) and rgb_img.shape[-1] > 3:
                        rgb_img = rgb_img[..., :3]
                    images.append(rgb_img)
                if self.include_depth and "depth" in cam_data:
                    images.append(cam_data["depth"])

            if len(images) > 0:
                images = torch.concat(images, axis=-1)

        # flatten the rest of the data which should just be state data
        if self.include_state and not (self.include_rgb or self.include_depth):
            if not self.include_oracle:
                observation.pop("oracle_info")
            else:
                observation = observation
        else:
            if not self.include_joints:
                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info']}
            else:
                # Create extra_agent dict with 'extra' and 'agent' keys
                extra_agent = {}
                for key in ['extra', 'agent']:
                    if key in observation:
                        extra_agent[key] = observation.pop(key)

                # Flatten the extra_agent dict
                extra_agent_flat = common.flatten_state_dict(extra_agent, use_torch=True, device=self.base_env.device)
                ret['joints'] = extra_agent_flat

                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info', 'extra']}

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

        if 'state' in ret.keys() and not self.include_state:
            ret.pop('state')

        if 'oracle_info' in ret.keys() and not self.include_oracle and ret['oracle_info'] is not None:
            ret.pop('oracle_info')

        if 'oracle_info' in ret.keys() and (ret['oracle_info'] == 4242424242).any().item():
            ret.pop('oracle_info')

        if 'prompt' in ret.keys() and (ret['prompt'] == 4242424242).any().item():
            ret.pop('prompt')

        if 'joints' in ret.keys() and not self.include_joints:
            ret.pop('joints')

        return ret


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 123
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill-MemoryBench"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `checkpoints/ppo_memtasks/runs/{run_name}/{TIME}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "ShellGamePush-v1"
    """the id of the environment"""
    include_state: bool = False
    """whether to include state information in observations"""
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024 # 512 | *256
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False # True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 90
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 270
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # ! 0.8 ! 
    """the discount factor gamma"""
    gae_lambda: float = 0.95 # ! 0.9 !
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32 # 32 | *8
    """the number of mini-batches"""
    update_epochs: int = 4 # 4 | *8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # ! False !
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""

    # ───── CAPS action-smoothness (Mysore et al. 2021) ─────
    # Pure actor-head loss regularizer; does NOT touch the surprise-memory
    # module (gradient flows only into actor_mean via detached state inputs).
    # Applied identically to all methods for a fair comparison.
    caps_lambda_t: float = 0.0
    """temporal smoothness weight: ||π_mean(s_t) − π_mean(s_{t+1})||² over consecutive steps"""
    caps_lambda_s: float = 0.0
    """spatial smoothness weight: ||π_mean(s) − π_mean(s+εσ)||² under input perturbation"""
    caps_sigma: float = 0.5
    """spatial-smoothness perturbation scale (× per-batch std of the fused state)"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = True

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


    include_oracle: bool = False
    """if toggled, oracle info (such as cup_with_ball_number in ShellGamePush-v0) will be used during the training, i.e. reducing memory task to MDP"""
    noop_steps: int = 1
    """if = 1, then no noops, if > 1, then noops for t ~ [0, noop_steps-1]"""
    include_rgb: bool = False
    """if toggled, rgb images will be included in the observation space"""
    include_joints: bool = False
    """[works only with include_rgb=True] if toggled, joints will be included in the observation space"""
    reward_mode: str = 'normalized_dense' # sparse | normalized_dense
    """the mode of the reward function"""
    
    # ───── EBM-specific args (replaces GRU args; kept under same names for
    # CLI compat where applicable, plus new ones) ─────
    saliency_ckpt: str = "analysis/ebm/path_a_head_v2.pt"
    """path to Path A v2 trained saliency head checkpoint"""
    no_saliency: bool = False
    """A3 ablation: ignore saliency head, push ALL DINOv2 patches into buffer
    (subject to L=64 cap and novelty filter). Tests H4 (filter is load-bearing)."""
    vit_backbone: str = "dinov2"
    """A_backbone ablation: 'dinov2' (default) or 'clip' for CLIP-ViT-B/16.
    Pair with the matching --saliency-ckpt (path_a_head_v3.pt vs path_a_head_v3_clip.pt)."""
    ebm_buffer_size: int = 64
    """episodic buffer length L"""
    ebm_top_k: int = 8
    """K candidate patches per step"""
    ebm_d_state: int = 256
    """fused state dim s_t"""
    ebm_novelty_thresh: float = 0.95
    """cosine similarity threshold for buffer novelty filter"""
    ebm_tau_age: float = 30.0
    """age decay constant for buffer eviction priority"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def print_tensor_shapes(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print_tensor_shapes(v, prefix=f'{prefix}{k}.')
        elif isinstance(v, torch.Tensor):
            print(f'{prefix}{k}: {v.shape}')


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        """
        oracle_info: dict with keys: "cup_with_ball_number" for ShellGame
        include_oracle: bool, if True, oracle_info will be used during the training, i.e. reducing memory task to MDP
        """
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256

        self.list_of_obs_keys = list(sample_obs.keys()) # 'oracle_info', 'prompt', 'state', 'rgb'

        if 'rgb' in self.list_of_obs_keys:
            in_channels = sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        for key in self.list_of_obs_keys:
            if key in ['oracle_info', 'prompt']:
                extractors[key] =  nn.Sequential(
                    nn.Linear(sample_obs[key].shape[-1], 64),
                    nn.ReLU()
                )
                self.out_features += 64
            elif key == 'joints':
                extractors[key] =  nn.Sequential(
                    nn.Linear(sample_obs[key].shape[-1], 256),
                    nn.ReLU()
                )
                self.out_features += 256

        print(f'{sample_obs.keys()=}')
        print_tensor_shapes(sample_obs)
        print('\n')

        # for state data we simply pass it through a single linear layer
        if 'state' in sample_obs.keys():
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb" and 'rgb' in self.list_of_obs_keys:
                obs = obs.float().permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)
                obs = obs / 255.
            elif key in ['oracle_info', 'prompt', 'joints']:
                obs = obs.float()

            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    """
    EBM-Robo agent. Replaces NatureCNN+GRU with EBMMemoryModule (frozen DINOv2 +
    frozen saliency head + episodic buffer + learned cross-attention reader).

    Interface mirrors the original GRU Agent:
      - get_action_and_value(obs, ebm_state, done, action=None)
      - get_action(obs, ebm_state, done, deterministic=False)
      - get_value(obs, ebm_state, done)

    But `ebm_state` here is NOT a tensor — it is a Python dict snapshot of the
    buffer (features/timestamps/saliency/used). At rollout the buffer is
    maintained inside `self.ebm`, so we ignore the passed-in `ebm_state` and
    use the live buffer. At update we restore from the cached snapshot before
    forwarding (managed by the outer training loop).
    """
    def __init__(self, envs, sample_obs):
        super().__init__()
        # EBM module — manages perception, buffer, reader, and fuse internally
        device = next(iter(self.parameters()), torch.tensor(0)).device  # placeholder
        proprio_dim = sample_obs["joints"].shape[-1] if "joints" in sample_obs else 25
        self.ebm = EBMMemoryModule(
            num_envs=args.num_envs,
            proprio_dim=proprio_dim,
            saliency_ckpt=args.saliency_ckpt,
            vit_backbone=args.vit_backbone,
            L=args.ebm_buffer_size,
            K=args.ebm_top_k,
            d_state=args.ebm_d_state,
            novelty_thresh=args.ebm_novelty_thresh,
            tau_age=args.ebm_tau_age,
            device="cuda",
            no_saliency=args.no_saliency,
        )
        d_state = args.ebm_d_state

        print('#'*50)
        print(f"EBM: num_envs={args.num_envs}, L={args.ebm_buffer_size}, "
              f"K={args.ebm_top_k}, d_state={d_state}")
        print('#'*50, '\n')

        self.critic = nn.Sequential(
            layer_init(nn.Linear(d_state, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(d_state, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)

        # rollout step counter (for buffer timestamping)
        self._t_global = 0

    def _step_ebm(self, x: dict, done: torch.Tensor, t_offset: int = 0) -> torch.Tensor:
        """
        x:    dict with 'rgb' (B, H, W, 6) and 'joints' (B, p)
        done: (B,) bool/float — reset buffer for envs where done==1 BEFORE forward.
        Returns s_t: (B, d_state).
        """
        if done is not None:
            self.ebm.reset(done.bool() if done.dtype != torch.bool else done)
        rgb = x["rgb"]
        proprio = x["joints"]
        s_t, _ = self.ebm.step(rgb, proprio, t=self._t_global + t_offset)
        return s_t

    def get_states(self, x: dict, ebm_state, done):
        """
        x: dict; if rgb has shape (T*B, H, W, 6) [from PPO update flat batch],
        we need to know T and B to walk through episode boundaries. The outer
        loop is responsible for restoring buffer state via self.ebm.restore()
        before calling this.

        For both rollout (single step) and update (replay) cases, we just call
        the EBM module forward — outer loop handles state restoration.
        """
        # rollout case: x has shape (B, H, W, 6); done is (B,)
        s_t = self._step_ebm(x, done)
        # The "next state" for the rollout-buffer logbook is just the buffer
        # snapshot post-step.
        return s_t, self.ebm.snapshot()

    def get_value(self, x, ebm_state, done):
        s_t, _ = self.get_states(x, ebm_state, done)
        return self.critic(s_t)

    def get_action(self, x, ebm_state, done, deterministic=False):
        s_t, new_state = self.get_states(x, ebm_state, done)
        action_mean = self.actor_mean(s_t)
        if deterministic:
            return action_mean, new_state
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample(), new_state

    def get_action_and_value(self, x, ebm_state, done, action=None):
        s_t, new_state = self.get_states(x, ebm_state, done)
        action_mean = self.actor_mean(s_t)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(s_t), new_state

    def reset_t_global(self, t: int = 0):
        self._t_global = t

    # ─── PPO update path: differentiable replay over cached buffer state ───

    def replay_action_value(self, cached_buffer: dict, cls_base, cls_hand, proprio,
                            gru_state_pre, gru_input, action=None):
        """V2-hybrid replay path. Inputs include the cached pre-step GRU state
        and the per-step gru_input recorded during rollout; the replay() call
        forwards GRU 1 step (differentiable) + reader + fuse.

        gru_state_pre: (mb_size, gru_hidden)
        gru_input:     (mb_size, gru_input_dim)
        """
        # ebm.replay expects gru_state_pre in (1, B, H) layout
        gru_state_pre_3d = gru_state_pre.unsqueeze(0).contiguous()
        s_t = self.ebm.replay(cached_buffer, cls_base, cls_hand, proprio,
                              gru_state_pre_3d, gru_input)
        action_mean = self.actor_mean(s_t)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (action,
                probs.log_prob(action).sum(1),
                probs.entropy().sum(1),
                self.critic(s_t),
                s_t)            # CAPS: expose fused state for action-smoothness reg



class AgentStateOnly(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.list_of_obs_keys = list(envs.single_observation_space.keys())
        print(f"{self.list_of_obs_keys=}")
        
        length = 0
        for key in self.list_of_obs_keys:
            l_ = np.array(envs.single_observation_space[key].shape).prod()
            print(f'{key}: {l_}')
            length += l_
        
        print(f'Total length: {length}')

        self.critic = nn.Sequential(
            layer_init(nn.Linear(length, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(length, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

        print(f'{envs.single_observation_space=}')
    
    def add_prompt_to_state(self, x):
        # Concatenate all observation tensors in order of self.list_of_obs_keys
        tensors = [x[key] for key in self.list_of_obs_keys]
        return torch.cat(tensors, dim=-1)

    def get_value(self, x):
        x = self.add_prompt_to_state(x)
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        x = self.add_prompt_to_state(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.add_prompt_to_state(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    TIME = time.strftime('%Y%m%d_%H%M%S')

    if args.env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
    elif args.env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                         'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                         'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
    elif args.env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}), # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")

    print('\n' + '='*75)
    print('║' + ' '*24 + 'Environment Configuration' + ' '*24 + '║')
    print('='*75)
    print('║' + f' Environment ID: {args.env_id}'.ljust(73) + '║')
    print('║' + f' Oracle Info:    {oracle_info}'.ljust(73) + '║')
    print('║ Wrappers:'.ljust(74) + '║')
    for wrapper, kwargs in wrappers_list:
        print('║    ├─ ' + wrapper.__name__.ljust(65) + '║')
        if kwargs:
            print('║    │  └─ ' + str(kwargs).ljust(65) + '║')
    print('║' + '-'*73 + '║')
    
    state_msg = 'state will be used' if args.include_state else 'state will not be used'
    print('║' + f' include_state:       {str(args.include_state):<5} │ {state_msg}'.ljust(68) + '║')
    
    rgb_msg = 'rgb images will be used' if args.include_rgb else 'rgb images will not be used'
    print('║' + f' include_rgb:         {str(args.include_rgb):<5} │ {rgb_msg}'.ljust(68) + '║')
    
    oracle_msg = 'oracle info will be used' if args.include_oracle else 'oracle info will not be used'
    print('║' + f' include_oracle:      {str(args.include_oracle):<5} │ {oracle_msg}'.ljust(68) + '║')
    
    joints_msg = 'joints will be used' if args.include_joints else 'joints will not be used'
    print('║' + f' include_joints:      {str(args.include_joints):<5} │ {joints_msg}'.ljust(68) + '║')
    print('='*75 + '\n')

    assert any([args.include_state, args.include_rgb]), "At least one of include_state or include_rgb must be True."
    assert not (args.include_joints and not args.include_rgb), "include_joints can only be True when include_rgb is True"

    if args.include_state and not args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'state'
    elif args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb'
    elif args.include_state and not args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_oracle is not implemented and does not make sense, since the state already contains oracle information")
        MODE = 'state_oracle'
    elif args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb_oracle is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb_oracle'
    elif not args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'rgb'
    elif not args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        MODE = 'rgb_oracle'
    elif not args.include_state and args.include_rgb and args.include_joints and args.include_oracle:
        MODE = 'rgb_joints_oracle' # TODO: check if this is correct
    elif not args.include_state and args.include_rgb and args.include_joints and not args.include_oracle:
        MODE = 'rgb_joints'
    else:
        raise NotImplementedError(f"Unknown mode: {args.include_state=} {args.include_rgb=} {args.include_oracle=} {args.include_joints=}")
    
    SAVE_DIR = f'checkpoints/ppo_memtasks/{MODE}/{args.reward_mode}/{args.env_id}'

    if 'state' in MODE:
        raise NotImplementedError("state mode is not implemented with GRU model, use rgb mode instead")


    print(f'{MODE=}')
    print(f'{prompt_info=}')

    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {})) # obs=torch.tensor -> dict with keys: state: obs, prompt: prompt, oracle_info: oracle_info


    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{MODE}__{TIME}"
    else:
        run_name = f"{args.exp_name}__{args.seed}__{MODE}__{TIME}"
    log_dir = f"{SAVE_DIR}/{run_name}/{TIME}"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    csv_fieldnames = [
        "iteration",
        "total_env_steps",
        "mode",
        "timestamp",
    ]
        
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if MODE not in ['state', 'state_oracle']:
        env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode=args.render_mode, sim_backend="gpu", reward_mode=args.reward_mode)
    else:
        env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode=args.render_mode, sim_backend="gpu", reward_mode=args.reward_mode) # render_mode="rgb_array",

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq,  **env_kwargs) # , reconfigure_freq=args.eval_reconfiguration_freq
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    for wrapper_class, wrapper_kwargs in wrappers_list:
        eval_envs = wrapper_class(eval_envs, **wrapper_kwargs)
        envs = wrapper_class(envs, **wrapper_kwargs)


    # DUAL-CAMERA: base + hand are concat'd channel-wise -> rgb shape (..., H, W, 6).
    # in_channels in the CNN is auto-detected from sample_obs["rgb"].shape[-1] = 6,
    # so no further conv-channel surgery is needed here.
    envs = FlattenRGBDObservationWrapperMulti(
        envs, rgb=args.include_rgb, depth=False, state=args.include_state,
        oracle=args.include_oracle, joints=args.include_joints,
        target_cameras=("base_camera", "hand_camera"),
    )
    eval_envs = FlattenRGBDObservationWrapperMulti(
        eval_envs, rgb=args.include_rgb, depth=False, state=args.include_state,
        oracle=args.include_oracle, joints=args.include_joints,
        target_cameras=("base_camera", "hand_camera"),
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"{SAVE_DIR}/{run_name}/{TIME}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"{SAVE_DIR}/{run_name}/{TIME}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    action_space_low = torch.tensor(envs.single_action_space.low, device=device, dtype=torch.float32)
    action_space_high = torch.tensor(envs.single_action_space.high, device=device, dtype=torch.float32)

    def clip_action(action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, action_space_low, action_space_high)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    print('='*70)
    print(f"Max Episode Steps: {max_episode_steps}")
    print('='*70 + '\n')
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"{SAVE_DIR}/{run_name}/{TIME}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)        
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    next_done_eval = torch.zeros(args.num_eval_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    video_iteration = 0

    print(f"\n####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####\n")

    if MODE not in ['state', 'state_oracle']:
        agent = Agent(envs, sample_obs=next_obs).to(device)
    else:
        agent = AgentStateOnly(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # ─── EBM rollout caches (per-step) ─────────────────────────────────────
    # We do NOT keep a "gru_states" tensor; instead per step we cache the
    # buffer snapshot (post-push state, what reader saw) plus the cls tokens
    # and proprio that built the query. PPO update replays reader+curr+fuse
    # over these caches WITHOUT re-running ViT / saliency head.
    d_vit  = agent.ebm.vit.dim
    L_buf  = agent.ebm.L
    p_dim  = agent.ebm.reader.W_Q.in_features - 128  # query_in = [proprio, curr(128)]

    cache_features   = torch.zeros(args.num_steps, args.num_envs, L_buf, d_vit, device=device)
    cache_used       = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long, device=device)
    cache_timestamps = torch.zeros(args.num_steps, args.num_envs, L_buf, dtype=torch.long, device=device)
    cache_saliency   = torch.full((args.num_steps, args.num_envs, L_buf), -1e9, device=device)
    cache_cls_base   = torch.zeros(args.num_steps, args.num_envs, d_vit, device=device)
    cache_cls_hand   = torch.zeros(args.num_steps, args.num_envs, d_vit, device=device)
    cache_proprio    = torch.zeros(args.num_steps, args.num_envs, p_dim, device=device)

    # V2-hybrid: cache GRU state PRE-step + the gru_input that step used.
    # Replay path forwards GRU 1 step from (cached_state, cached_input) so
    # gradients flow through gru params.
    gru_hidden = agent.ebm.gru_state.shape[-1]  # auto: H for GRU, 2H for LSTM
    gru_in_dim = agent.ebm.gru_input_dim
    cache_gru_state_pre = torch.zeros(args.num_steps, args.num_envs, gru_hidden, device=device)
    cache_gru_input     = torch.zeros(args.num_steps, args.num_envs, gru_in_dim, device=device)

    # Eval needs its own EBM with num_envs = num_eval_envs but shared params.
    # Lightweight: instantiate a fresh EBMMemoryModule and overwrite its
    # learned submodules with references to the train-side ones.
    agent_eval_ebm = EBMMemoryModule(
        num_envs=args.num_eval_envs,
        proprio_dim=p_dim,
        saliency_ckpt=args.saliency_ckpt,
        vit_backbone=args.vit_backbone,
        L=args.ebm_buffer_size,
        K=args.ebm_top_k,
        d_state=args.ebm_d_state,
        novelty_thresh=args.ebm_novelty_thresh,
        tau_age=args.ebm_tau_age,
        device="cuda",
        no_saliency=args.no_saliency,
    )
    # Share frozen + learned submodules with the training EBM (params not
    # duplicated; only the buffer is independent).
    agent_eval_ebm.vit            = agent.ebm.vit
    # SRB has no saliency_head / xy_concat (whole point of the module).
    agent_eval_ebm.curr_summary   = agent.ebm.curr_summary
    agent_eval_ebm.reader         = agent.ebm.reader
    agent_eval_ebm.fuse           = agent.ebm.fuse
    # NoLSTM ablation: there is no recurrent branch to share.

    # for iteration in range(1, args.num_iterations + 1):
    for iteration in tqdm(range(1, args.num_iterations + 1), total=args.num_iterations, desc="Training"):
        print(f"Epoch: {iteration}, global_step={global_step}")
        # reset the training EBM buffer for ALL envs at the start of each
        # iteration (each task has max_episode_steps == args.num_steps so a
        # rollout iteration covers exactly one episode per env).
        agent.ebm.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))
        agent.reset_t_global(0)
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            agent_eval_ebm.reset(torch.ones(args.num_eval_envs, dtype=torch.bool, device=device))
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            eval_t = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    # reset eval buffer for envs whose previous episode just ended
                    agent_eval_ebm.reset(next_done_eval.bool())
                    s_t_eval, _ = agent_eval_ebm.step(eval_obs["rgb"], eval_obs["joints"], t=eval_t)
                    act_eval = agent.actor_mean(s_t_eval)  # deterministic = mean
                    eval_t += 1
                    act_eval = clip_action(act_eval)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(act_eval)
                    next_done_eval = torch.logical_or(eval_terminations, eval_truncations).to(torch.float32)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            csv_row = {
                "iteration": iteration,
                "total_env_steps": global_step,
                "mode": "eval",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"{Fore.GREEN}Evaluation Metric: {k}{Style.RESET_ALL} | {Fore.CYAN}Mean: {mean:.4f}{Style.RESET_ALL}")
                csv_row[k] = round(mean.item(), 4)
                if k not in csv_fieldnames:
                    csv_fieldnames.append(k)
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"{SAVE_DIR}/{run_name}/{TIME}/ckpt_{video_iteration}_{iteration}.pt"
            video_iteration += 1
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # Reset EBM buffer for envs that just terminated (next_done is the
            # done flag at the START of this step — those envs are starting
            # a fresh episode here).
            with torch.no_grad():
                agent.ebm.reset(next_done.bool())
                # V2-hybrid: snapshot gru_state BEFORE this step (used as initial
                # state in replay path).
                cache_gru_state_pre[step] = agent.ebm.gru_state.squeeze(0).detach().clone()
                # forward EBM step (this consumes gru_state and returns the new one + gru_input that was fed)
                s_t, info = agent.ebm.step(next_obs["rgb"], next_obs["joints"], t=step)

                # cache POST-PUSH buffer state (= what reader saw)
                cache_features[step]   = agent.ebm.buffer.features.detach()
                cache_used[step]       = agent.ebm.buffer.used.clone()
                cache_timestamps[step] = agent.ebm.buffer.timestamps.clone()
                cache_saliency[step]   = agent.ebm.buffer.saliency.clone()
                cache_cls_base[step]   = info["cls_base"]
                cache_cls_hand[step]   = info["cls_hand"]
                cache_proprio[step]    = next_obs["joints"].detach()
                cache_gru_input[step]  = info["gru_input"]

                # actor / critic on s_t
                action_mean = agent.actor_mean(s_t)
                action_logstd = agent.actor_logstd.expand_as(action_mean)
                std = torch.exp(action_logstd)
                probs = Normal(action_mean, std)
                action = probs.sample()
                logprob = probs.log_prob(action).sum(-1)
                value = agent.critic(s_t)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                csv_row = {
                    "iteration": iteration,
                    "total_env_steps": global_step,
                    "mode": "train",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                for k, v in final_info["episode"].items():
                    sliced = v[done_mask].float()
                    if sliced.numel() > 0:
                        mean_value = sliced.mean()
                        if logger is not None:
                            logger.add_scalar(f"train/{k}", mean_value, global_step)
                        csv_row[k] = round(mean_value.item(), 4)
                        if k not in csv_fieldnames:
                            csv_fieldnames.append(k)
                file_exists = os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(csv_row)
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    # snapshot/restore EBM so bootstrap doesn't perturb buffer
                    _snap = agent.ebm.snapshot()
                    fv = agent.get_value(infos["final_observation"], None, next_done).view(-1)
                    agent.ebm.restore(_snap)
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = fv[done_mask]

        rollout_time = time.time() - rollout_time

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            _snap = agent.ebm.snapshot()
            next_value = agent.get_value(next_obs, None, next_done).reshape(1, -1)
            agent.ebm.restore(_snap)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))   # kept for compatibility; not used in update
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)
        # flatten EBM caches so we can index by mb_inds
        b_features   = cache_features.reshape(-1, L_buf, d_vit)
        b_used       = cache_used.reshape(-1)
        b_timestamps = cache_timestamps.reshape(-1, L_buf)
        b_saliency   = cache_saliency.reshape(-1, L_buf)
        b_cls_base   = cache_cls_base.reshape(-1, d_vit)
        b_cls_hand   = cache_cls_hand.reshape(-1, d_vit)
        b_proprio    = cache_proprio.reshape(-1, p_dim)
        # V2-hybrid: cached GRU pre-step state and per-step gru input
        b_gru_state_pre = cache_gru_state_pre.reshape(-1, gru_hidden)
        b_gru_input     = cache_gru_input.reshape(-1, gru_in_dim)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # V2-hybrid: replay reader+curr+GRU+fuse on cached state for this minibatch.
                cached = {
                    "features":   b_features[mb_inds],
                    "used":       b_used[mb_inds],
                    "timestamps": b_timestamps[mb_inds],
                    "saliency":   b_saliency[mb_inds],
                }
                _, newlogprob, entropy, newvalue, s_t_mb = agent.replay_action_value(
                    cached,
                    b_cls_base[mb_inds],
                    b_cls_hand[mb_inds],
                    b_proprio[mb_inds],
                    b_gru_state_pre[mb_inds],
                    b_gru_input[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # ── CAPS action-smoothness (actor-head only; memory detached) ──
                # Gradients flow ONLY into actor_mean — the surprise-memory module
                # (buffer/reader/fuse/LSTM) is never touched by CAPS, so the
                # main method's architecture & narrative are unchanged.
                caps_loss = s_t_mb.new_zeros(())
                if args.caps_lambda_t > 0.0:
                    mb_t = torch.as_tensor(mb_inds, device=device, dtype=torch.long)
                    nxt = mb_t + args.num_envs                       # same env, t+1
                    valid = mb_t < (args.num_steps - 1) * args.num_envs
                    # exclude pairs that straddle an episode reset
                    valid = valid & (b_dones[nxt.clamp(max=args.batch_size - 1)] < 0.5)
                    if valid.any():
                        nv = nxt[valid]
                        with torch.no_grad():
                            *_, s_t_next = agent.replay_action_value(
                                {"features": b_features[nv], "used": b_used[nv],
                                 "timestamps": b_timestamps[nv], "saliency": b_saliency[nv]},
                                b_cls_base[nv], b_cls_hand[nv], b_proprio[nv],
                                b_gru_state_pre[nv], b_gru_input[nv], None)
                        a_t = agent.actor_mean(s_t_mb[valid].detach())
                        a_n = agent.actor_mean(s_t_next.detach())
                        caps_loss = caps_loss + args.caps_lambda_t * ((a_t - a_n) ** 2).sum(-1).mean()
                if args.caps_lambda_s > 0.0:
                    s = s_t_mb.detach()
                    sig = args.caps_sigma * (s.std() + 1e-6)
                    a0 = agent.actor_mean(s)
                    a1 = agent.actor_mean(s + sig * torch.randn_like(s))
                    caps_loss = caps_loss + args.caps_lambda_s * ((a0 - a1) ** 2).sum(-1).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + caps_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("charts/global_step", global_step, global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/caps", float(caps_loss), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

    if args.save_model and not args.evaluate:
        model_path = f"{SAVE_DIR}/{run_name}/{TIME}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    if logger is not None: 
        logger.close()