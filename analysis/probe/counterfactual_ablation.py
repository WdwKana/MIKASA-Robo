"""
F. Counterfactual ablation: intervene on the recurrent hidden output (gru_hidden
or LSTM h-state) at eval time, measure SR drop.

Tests CAUSAL claim "the recurrent state's contribution to the actor input is
what gives V2/V3a their SR_end advantage over V1." If ablating gru_hidden
(setting it to zero or shuffling it across envs) drops SR_end down to V1
level (~0.10), the claim holds. If SR remains high, the recurrent path is
NOT causally responsible and we have to look elsewhere.

Conditions per method:
  - baseline       : no intervention (sanity, should match training eval)
  - zero_hidden    : gru_hidden set to 0 before fuse (kills its contribution)
  - shuffle_hidden : per-env shuffle of gru_hidden (preserves marginal stats,
                     breaks the "this env's history" identity)
  - random_hidden  : replace with N(0, 1) noise

For V1 (no recurrent state): only baseline.
For V2 (GRU): intervene on gru_hidden = gru_state.squeeze(0).
For V3a (LSTM): intervene on h-part of gru_state before it enters fuse.
For pure-LSTM: intervene on lstm hidden output (which IS s_t).

Number of episodes per (method, seed, condition) is configurable; default 32
gives ~0.06 standard error on a binary SR_end estimate.
"""
from __future__ import annotations
import argparse, json, sys
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
from mikasa_robo_suite.utils.wrappers import *
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti
from baselines.ppo.modules import (
    EBMMemoryModule, EBMHybridMemoryModule, EBMHybridLSTMMemoryModule,
)

DEVICE = torch.device("cuda")


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
                                             oracle=False, joints=True,
                                             target_cameras=("base_camera","hand_camera"))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


METHOD_MODULES = {"v1": EBMMemoryModule, "v2": EBMHybridMemoryModule, "v3a": EBMHybridLSTMMemoryModule}


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


def intervene(gru_hidden: torch.Tensor, condition: str) -> torch.Tensor:
    if condition == "baseline":
        return gru_hidden
    if condition == "zero_hidden":
        return torch.zeros_like(gru_hidden)
    if condition == "shuffle_hidden":
        idx = torch.randperm(gru_hidden.shape[0], device=gru_hidden.device)
        return gru_hidden[idx]
    if condition == "random_hidden":
        return torch.randn_like(gru_hidden)
    raise ValueError(condition)


@torch.no_grad()
def custom_step(method: str, ebm, rgb6, proprio, t: int, condition: str):
    """Replicates ebm.step() but injects intervention right before fuse."""
    tok_b, tok_h, cls_b, cls_h = ebm.vit(rgb6)
    tokens_all = torch.cat([tok_b, tok_h], dim=1)
    B = tokens_all.size(0)

    if ebm.no_saliency:
        idx = torch.randint(0, tokens_all.size(1), (B, ebm.K), device=tokens_all.device)
        cand_feats = tokens_all.gather(1, idx.unsqueeze(-1).expand(B, ebm.K, tokens_all.size(-1)))
        cand_sal = torch.ones(B, ebm.K, device=tokens_all.device)
    else:
        logits = ebm.saliency_head(tokens_all, ebm.xy_concat)
        probs = torch.sigmoid(logits)
        topk_val, topk_idx = probs.topk(ebm.K, dim=-1)
        cand_feats = tokens_all.gather(1, topk_idx.unsqueeze(-1).expand(B, ebm.K, tokens_all.size(-1)))
        cand_sal = topk_val * (topk_val > 0.5).float()

    ebm.buffer.push_batch(cand_feats, cand_sal, t_now=t)

    curr = ebm.curr_summary(torch.cat([cls_b, cls_h], dim=-1))

    if method == "v1":
        feats, mask, ts, sal = ebm.buffer.get()
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = ebm.reader(query_in, feats, mask, ts, sal)
        s_t = ebm.fuse(torch.cat([proprio, curr, retrieved], dim=-1))
        return s_t

    # V2 / V3a: recurrent path
    pool_w = torch.softmax(cand_sal, dim=-1).unsqueeze(-1)
    pooled = (pool_w * cand_feats).sum(dim=1)
    gru_input = torch.cat([pooled, curr, proprio], dim=-1).unsqueeze(0)

    if method == "v2":
        new_state, _ = ebm.gru(gru_input, ebm.gru_state)
        ebm.gru_state = new_state.detach()
        gru_hidden = ebm.gru_state.squeeze(0)
    elif method == "v3a":
        h, c = ebm._split_state(ebm.gru_state)
        _, (new_h, new_c) = ebm.lstm(gru_input, (h, c))
        ebm.gru_state = ebm._join_state(new_h, new_c).detach()
        gru_hidden = new_h.squeeze(0)
    else:
        raise ValueError(method)

    # ---- INTERVENTION POINT ----
    gru_hidden = intervene(gru_hidden, condition)

    feats, mask, ts, sal = ebm.buffer.get()
    query_in = torch.cat([proprio, curr], dim=-1)
    retrieved = ebm.reader(query_in, feats, mask, ts, sal)
    s_t = ebm.fuse(torch.cat([proprio, curr, retrieved, gru_hidden], dim=-1))
    return s_t


@torch.no_grad()
def eval_with_intervention(method, ck, env, num_episodes, num_envs, condition):
    obs, _ = env.reset(seed=12345)
    agent = make_agent(method, ck, env, obs, num_envs)
    agent.ebm.reset(torch.ones(num_envs, dtype=torch.bool, device=DEVICE))
    sr_once_list, sr_end_list = [], []
    done_prev = torch.zeros(num_envs, device=DEVICE)
    eps = 0; t = 0
    while eps < num_episodes:
        agent.ebm.reset(done_prev.bool())
        s_t = custom_step(method, agent.ebm, obs["rgb"], obs["joints"], t, condition)
        act = agent.actor_mean(s_t)
        obs, _, term, trunc, infos = env.step(act)
        done = torch.logical_or(term, trunc)
        if "final_info" in infos:
            mask = infos["_final_info"]
            ep_info = infos["final_info"]["episode"]
            once = ep_info.get("success_once")
            end_ = ep_info.get("success_at_end")
            for e in range(num_envs):
                if not bool(mask[e].item()): continue
                if once is not None: sr_once_list.append(float(once[e].item()))
                if end_ is not None: sr_end_list.append(float(end_[e].item()))
                eps += 1
        done_prev = done.float()
        t = (t + 1) % 1000
    return {
        "sr_once_mean": float(np.mean(sr_once_list)) if sr_once_list else float("nan"),
        "sr_end_mean":  float(np.mean(sr_end_list)) if sr_end_list else float("nan"),
        "sr_once_std":  float(np.std(sr_once_list, ddof=1)) if len(sr_once_list)>1 else 0.0,
        "sr_end_std":   float(np.std(sr_end_list, ddof=1)) if len(sr_end_list)>1 else 0.0,
        "n_episodes":   int(eps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["v1","v2","v3a"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="InterceptMedium-v0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episodes", type=int, default=32)
    p.add_argument("--conditions", nargs="+",
                   default=["baseline","zero_hidden","shuffle_hidden","random_hidden"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.method == "v1":
        args.conditions = ["baseline"]   # V1 has no recurrent state

    results = {}
    for cond in args.conditions:
        print(f"[F] {args.method} cond={cond}")
        env = build_env(args.env_id, args.num_envs)
        r = eval_with_intervention(args.method, args.checkpoint, env, args.num_episodes,
                                   args.num_envs, cond)
        results[cond] = r
        print(f"   SR_once={r['sr_once_mean']:.3f}±{r['sr_once_std']:.3f}  "
              f"SR_end={r['sr_end_mean']:.3f}±{r['sr_end_std']:.3f}  "
              f"n_ep={r['n_episodes']}")
        del env
        torch.cuda.empty_cache()

    out = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "num_episodes_per_condition": args.num_episodes,
        "results": results,
    }
    out_path = args.output or str(Path(args.checkpoint).parent / "F_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[F] saved to {out_path}")


if __name__ == "__main__":
    main()
