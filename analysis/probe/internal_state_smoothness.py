"""Phase 1: Use existing probe2 data to test sub-claims A and E.

A) V3a's s_t is more temporally smooth than V1/V2/V3a near ball
E) Smoothness drop is monotonic across distance bins (not single-threshold artifact)

We don't have access to the raw arrays inside probe2 results (only summaries),
so we need to re-collect briefly per method.
"""
import sys
sys.path.insert(0, '/zfsstore/user/s4176650/MIKASA-Robo')
import json, numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs import *  # noqa
from mikasa_robo_suite.utils.wrappers import *
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
from baselines.ppo.modules import (
    EBMMemoryModule, EBMHybridMemoryModule, EBMHybridLSTMMemoryModule,
)

DEVICE = torch.device("cuda")
ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
METHOD_MODULES = {"v1": EBMMemoryModule, "v2": EBMHybridMemoryModule, "v3a": EBMHybridLSTMMemoryModule}


def build_env(env_id, num_envs):
    env = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   render_mode="all", sim_backend="gpu", reward_mode="normalized_dense")
    for W, kw in [(StateOnlyTensorToDictWrapper,{}),
                  (InitialZeroActionWrapper,{"n_initial_steps":0}),
                  (RenderStepInfoWrapper,{}),(RenderRewardInfoWrapper,{}),
                  (DebugRewardWrapper,{})]:
        env = W(env, **kw)
    env = FlattenRGBDObservationWrapperMulti(env, rgb=True, depth=False, state=False,
        oracle=False, joints=True, target_cameras=("base_camera","hand_camera"))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def make_agent(method, ck, envs, sample_obs, num_envs):
    ModCls = METHOD_MODULES[method]
    pd = sample_obs["joints"].shape[-1]
    ebm = ModCls(num_envs=num_envs, proprio_dim=pd,
                 saliency_ckpt=str(ROOT/"analysis/ebm/path_a_head_v3.pt"),
                 vit_backbone="dinov2", L=64, K=8, d_state=256,
                 novelty_thresh=0.95, tau_age=30.0, device="cuda", no_saliency=False)
    class A(nn.Module):
        def __init__(s):
            super().__init__(); s.ebm = ebm
            from baselines.ppo.ppo_memtasks_ebm import layer_init
            act_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
            s.critic = nn.Sequential(layer_init(nn.Linear(256,512)),nn.ReLU(),layer_init(nn.Linear(512,1)))
            s.actor_mean = nn.Sequential(layer_init(nn.Linear(256,512)),nn.ReLU(),
                layer_init(nn.Linear(512,act_dim),std=0.01*np.sqrt(2)),nn.Tanh())
            s.actor_logstd = nn.Parameter(torch.ones(1,act_dim)*-0.5)
    agent = A().to(DEVICE)
    sd = torch.load(ck, map_location=DEVICE)
    own = {k:v.shape for k,v in agent.state_dict().items()}
    filt = {k:v for k,v in sd.items() if k not in own or v.shape == own[k]}
    agent.load_state_dict(filt, strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def collect(agent, env, num_episodes, num_envs):
    obs, _ = env.reset(seed=12345)
    agent.ebm.reset(torch.ones(num_envs, dtype=torch.bool, device=DEVICE))
    pending = [[] for _ in range(num_envs)]
    out = {"s":[], "a":[], "t2b":[], "step":[], "ep":[], "succ":[]}
    ep_id = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    step_in_ep = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    done_prev = torch.zeros(num_envs, device=DEVICE)
    eps_done = 0; t = 0
    while eps_done < num_episodes:
        agent.ebm.reset(done_prev.bool())
        s_t, _ = agent.ebm.step(obs["rgb"], obs["joints"], t=t)
        inner = env._env.unwrapped
        t2b = (inner.ball.pose.p - inner.agent.tcp.pose.p).detach().cpu().numpy()
        act = agent.actor_mean(s_t)
        s_np = s_t.detach().cpu().numpy(); a_np = act.detach().cpu().numpy()
        step_np = step_in_ep.cpu().numpy(); ep_np = ep_id.cpu().numpy()
        for e in range(num_envs):
            pending[e].append((s_np[e], a_np[e], t2b[e], int(step_np[e]), int(ep_np[e])))
        obs, _, term, trunc, infos = env.step(act)
        done = torch.logical_or(term, trunc)
        if "final_info" in infos:
            done_mask = infos["_final_info"]
            sr_end = infos["final_info"]["episode"].get("success_at_end")
            for e in range(num_envs):
                if not bool(done_mask[e].item()): continue
                is_succ = float(sr_end[e].item()) if sr_end is not None else 0.0
                for rec in pending[e]:
                    out["s"].append(rec[0]); out["a"].append(rec[1])
                    out["t2b"].append(rec[2]); out["step"].append(rec[3])
                    out["ep"].append(rec[4]); out["succ"].append(is_succ)
                pending[e] = []; ep_id[e] += 1; eps_done += 1
        done_prev = done.float()
        step_in_ep = torch.where(done, torch.zeros_like(step_in_ep), step_in_ep + 1)
        t = (t + 1) % 1000
    for k in out: out[k] = np.array(out[k])
    return out


def delta_sq_per_step(arr, ep):
    """Compute ‖x_t - x_{t-1}‖² per step (NaN at episode boundaries)."""
    d = np.full(arr.shape[0], np.nan)
    for i in range(1, arr.shape[0]):
        if ep[i] != ep[i-1]: continue
        d[i] = np.sum((arr[i] - arr[i-1])**2)
    return d


def analyze(data, label):
    s = data["s"]; a = data["a"]; t2b = data["t2b"]
    dist = np.linalg.norm(t2b, axis=1)
    ep = data["ep"]; succ = data["succ"]
    s_d = delta_sq_per_step(s, ep)
    a_d = delta_sq_per_step(a, ep)

    # Distance bins
    bins = [0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 5.00]
    bin_labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins)-1)]
    bin_idx = np.digitize(dist, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins)-2)

    print(f"\n=== {label} ===")
    print(f"{'dist_bin':<18} {'N':<7} {'s_t_Δ²':<12} {'a_t_Δ²':<12} {'corr(s,a)':<10}")
    for b in range(len(bins)-1):
        mask = (bin_idx == b) & ~np.isnan(s_d) & ~np.isnan(a_d)
        n = int(mask.sum())
        if n < 10:
            print(f"{bin_labels[b]:<18} {n:<7} -            -            -")
            continue
        s_m = s_d[mask].mean(); a_m = a_d[mask].mean()
        # correlation between s-delta and a-delta within bin
        try:
            corr = np.corrcoef(s_d[mask], a_d[mask])[0,1]
        except Exception:
            corr = np.nan
        print(f"{bin_labels[b]:<18} {n:<7} {s_m:<12.4f} {a_m:<12.4f} {corr:<10.3f}")

    # Overall corr
    mask = ~np.isnan(s_d) & ~np.isnan(a_d)
    overall_corr = np.corrcoef(s_d[mask], a_d[mask])[0,1]
    print(f"OVERALL corr(‖s_d‖²,‖a_d‖²) = {overall_corr:.3f}")


def main():
    import sys
    METHODS_SEEDS = [("v1", 42), ("v2", 42), ("v3a", 42)]
    PREFIXES = {"v1": "ppo-ebm-imed-", "v2": "ppo-ebm-hybrid-imed-", "v3a": "ppo-v3a-lstm-imed-"}
    ROOT_TASK = ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptMedium-v0"
    NUM_ENVS = 16; NUM_EPISODES = 32

    for method, seed in METHODS_SEEDS:
        dirs = sorted(ROOT_TASK.glob(f"{PREFIXES[method]}seed{seed}__*"))
        d = dirs[-1]
        ck = d / d.name.split("__")[-1] / "final_ckpt.pt"
        ck_str = str(ck)
        # find correct ck — name contains timestamp
        ck_candidates = list(d.rglob("final_ckpt.pt"))
        ck_str = str(ck_candidates[0])
        env = build_env("InterceptMedium-v0", NUM_ENVS)
        obs, _ = env.reset(seed=0)
        agent = make_agent(method, ck_str, env, obs, NUM_ENVS)
        print(f"\n[collect] {method} seed={seed} ckpt={ck_str}")
        data = collect(agent, env, NUM_EPISODES, NUM_ENVS)
        print(f"  collected: {data['s'].shape[0]} steps")
        analyze(data, f"{method} seed={seed}")
        del agent, env
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
