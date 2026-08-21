"""Exp 2 (SRB-TR variant) — closed-loop rollout collector for the SRB-TR agent.

Records, per step:
    s      (T, N, d_state)  reader output (what the policy conditions on)
    bufmean(T, N, d_vit)    masked mean of episodic-buffer features (probe target)
    surp   (T, N)           max surprise score this step
    npush  (T, N)           number of tokens written this step
    succ   (T, N)           env success flag
    color  (N,)             ground-truth cue color
Final-step buffer snapshot per batch:
    buf_feats (N, L, d_vit), buf_mask (N, L)

Usage mirrors exp2_collect_rollouts.py but for the srbtr crescaps entry.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

REPO = "/zfsstore/user/s4176650/MIKASA-Robo"
sys.path.insert(0, REPO)

import numpy as np
import torch

from exp2_collect_rollouts import build_eval_envs, build_intercept_envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="RememberColor5-v0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-ball", action="store_true",
                    help="Intercept: log ball pos/vel + LSTM-branch state + frame feature")
    ap.add_argument("--entry", choices=["cres_caps", "plain"], default="cres_caps",
                    help="which srbtr training entry the checkpoint came from")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")

    entry_mod = ("baselines.ppo.ppo_memtasks_ebm_srb_tr_cres_caps"
                 if a.entry == "cres_caps" else "baselines.ppo.ppo_memtasks_ebm_srb_tr")
    M = importlib.import_module(entry_mod)
    M.args = M.Args(env_id=a.env_id, include_rgb=True, include_joints=True,
                    num_envs=a.num_envs, capture_video=False, save_model=False)

    base_env = None
    if a.record_ball:
        envs, base_env = build_intercept_envs(a.env_id, a.num_envs)
    else:
        envs = build_eval_envs(M, a.env_id, a.num_envs, a.seed)
    obs, info = envs.reset(seed=a.seed)

    agent = M.Agent(envs, sample_obs=obs).to(device)
    # drop per-env runtime buffers saved at training num_envs (reset at rollout anyway)
    ckpt_sd = torch.load(a.ckpt, map_location=device)
    own_sd = agent.state_dict()
    filtered = {k: v for k, v in ckpt_sd.items()
                if k in own_sd and own_sd[k].shape == v.shape}
    dropped = sorted(set(ckpt_sd) - set(filtered))
    if dropped:
        print(f"dropped {len(dropped)} runtime-state keys: {dropped[:6]}...")
    for k in dropped:  # every dropped key must be per-env runtime state, not weights
        assert ckpt_sd[k].shape[0] in (1, 256) and "weight" not in k and "bias" not in k, \
            f"refusing to drop learned param {k} {tuple(ckpt_sd[k].shape)}"
    missing, unexpected = agent.load_state_dict(filtered, strict=False)
    unexplained = [k for k in missing if k not in dropped]
    assert not unexplained, f"missing learned params: {unexplained}"
    agent.eval()

    # frame-level control feature for the tracking probe: mean-pooled ViT tokens
    vit_slot = {}
    if a.record_ball:
        agent.ebm.vit.register_forward_hook(
            lambda mod, inp, out: vit_slot.__setitem__(
                "v", torch.cat([out[0], out[1]], dim=1).mean(1).detach()))

    N, T = a.num_envs, a.horizon
    for b in range(a.batches):
        obs, info = envs.reset(seed=a.seed + 1000 * (b + 1))
        color = (info["oracle_info"].detach().cpu().numpy().astype(np.int64)
                 if not a.record_ball else np.zeros(N, dtype=np.int64))
        agent.ebm.reset(torch.ones(N, dtype=torch.bool, device=device))
        agent._t_global = 0

        s_log = surp_log = npush_log = bufmean_log = reader_log = None
        succ_log = np.zeros((T, N), dtype=np.bool_)
        wm_log = emb_log = ball_pos_log = ball_vel_log = None
        if a.record_ball:
            wm_dim = agent.ebm.gru_state.shape[-1]
            wm_log = np.zeros((T, N, wm_dim), dtype=np.float16)
            ball_pos_log = np.zeros((T, N, 3), dtype=np.float32)
            ball_vel_log = np.zeros((T, N, 3), dtype=np.float32)

        for t in range(T):
            with torch.no_grad():
                rgb = obs["rgb"]
                proprio = obs["joints"]
                s_t, step_info = agent.ebm.step(rgb, proprio, t=t)
                action = agent.actor_mean(s_t)  # deterministic
            retrieved = step_info["retrieved"]
            if s_log is None:
                s_log = np.zeros((T, N, s_t.shape[-1]), dtype=np.float16)
                reader_log = np.zeros((T, N, retrieved.shape[-1]), dtype=np.float16)
                surp_log = np.zeros((T, N), dtype=np.float32)
                npush_log = np.zeros((T, N), dtype=np.int16)
                d_vit = agent.ebm.buffer.features.shape[-1]
                bufmean_log = np.zeros((T, N, d_vit), dtype=np.float16)
            s_log[t] = s_t.float().cpu().numpy()
            reader_log[t] = retrieved.float().cpu().numpy()
            surp_log[t] = step_info["max_surprise"].float().cpu().numpy()
            npush_log[t] = step_info["n_pushed"].cpu().numpy()
            feats, mask = agent.ebm.buffer.features, agent.ebm.buffer.mask
            denom = mask.sum(-1, keepdim=True).clamp(min=1).float()
            bufmean_log[t] = ((feats * mask.unsqueeze(-1)).sum(1) / denom).float().cpu().numpy()
            if a.record_ball:
                wm_log[t] = agent.ebm.gru_state[0].float().cpu().numpy()
                if emb_log is None:
                    emb_log = np.zeros((T, N, vit_slot["v"].shape[-1]), dtype=np.float16)
                emb_log[t] = vit_slot["v"].float().cpu().numpy()
                ball_pos_log[t] = base_env.ball.pose.p.cpu().numpy()
                ball_vel_log[t] = base_env.ball.linear_velocity.cpu().numpy()

            obs, rew, term, trunc, info = envs.step(action)
            succ_log[t] = info["success"].detach().cpu().numpy()

        extra = {}
        if a.record_ball:
            extra = dict(h=wm_log, emb=emb_log,  # exp4-compatible keys
                         ball_pos=ball_pos_log, ball_vel=ball_vel_log)
        np.savez_compressed(
            os.path.join(a.out, f"batch{b:03d}.npz"),
            s=s_log, bufmean=bufmean_log, reader=reader_log,
            surp=surp_log, npush=npush_log,
            succ=succ_log, color=color,
            buf_feats=agent.ebm.buffer.features.float().cpu().numpy().astype(np.float16),
            buf_mask=agent.ebm.buffer.mask.cpu().numpy(), **extra)
        print(f"batch {b + 1}/{a.batches}  succ_once={succ_log.any(0).mean():.2f}  "
              f"succ_end={succ_log[-1].mean():.2f}", flush=True)

    envs.close()


if __name__ == "__main__":
    main()
