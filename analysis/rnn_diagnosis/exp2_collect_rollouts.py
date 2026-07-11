"""Exp 2 — closed-loop rollout collector for memory probing.

Rolls a trained crescaps agent (lstm / gru) in its eval environment and records,
per step: the recurrent hidden state, the encoder embedding (probe ceiling /
behavioural-leakage control), env success, plus the episode's ground-truth cue
color. Fixed 60-step episodes (ignore_terminations), deterministic actions —
identical to the training script's eval protocol.

Output: one npz per batch under --out:
    h      (T, N, H)   recurrent hidden state after consuming obs_t
    c      (T, N, H)   LSTM cell state (absent for gru)
    emb    (T, N, D)   feature_net output (DINOv2+CRES encoder, pre-RNN)
    succ   (T, N)      per-step success flag
    color  (N,)        ground-truth cue color index
Usage:
    python exp2_collect_rollouts.py --method lstm --env-id RememberColor5-v0 \
        --ckpt <path.pt> --batches 20 --out out/rollouts_lstm_rc5_seed33
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

ENTRY = {
    "lstm": "baselines.ppo.ppo_memtasks_dinov2_lstm_cres_caps",
    "gru": "baselines.ppo.ppo_memtasks_dinov2_gru_cres_caps",
    # single entry with flags; crescaps config = color_aug=True
    "ffm": "baselines.ppo.ppo_memtasks_dinov2_ffm",
}
EXTRA_ARGS = {"ffm": dict(color_aug=True)}


def build_eval_envs(M, env_id: str, num_envs: int, seed: int):
    import gymnasium as gym
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mikasa_robo_suite.utils.wrappers import (
        DebugRewardWrapper,
        InitialZeroActionWrapper,
        RememberColorInfoWrapper,
        RememberShapeInfoWrapper,
        RenderRewardInfoWrapper,
        RenderStepInfoWrapper,
        StateOnlyTensorToDictWrapper,
    )
    from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti

    env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos",
                      render_mode="sensors", sim_backend="gpu",
                      reward_mode="normalized_dense")
    envs = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)

    info_wrapper = (RememberShapeInfoWrapper if "RememberShape" in env_id
                    else RememberColorInfoWrapper)
    for wrapper_class, kw in [
        (StateOnlyTensorToDictWrapper, {}),
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (info_wrapper, {}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
        (DebugRewardWrapper, {}),
    ]:
        envs = wrapper_class(envs, **kw)
    envs = FlattenRGBDObservationWrapperMulti(
        envs, rgb=True, depth=False, state=False, oracle=False, joints=True,
        target_cameras=("base_camera", "hand_camera"))
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True,
                              record_metrics=True)
    return envs


def build_intercept_envs(env_id: str, num_envs: int):
    """Intercept variant: no task info wrapper; returns (vec_envs, base_env)."""
    import gymnasium as gym
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mikasa_robo_suite.utils.wrappers import (
        DebugRewardWrapper,
        InitialZeroActionWrapper,
        RenderRewardInfoWrapper,
        RenderStepInfoWrapper,
        StateOnlyTensorToDictWrapper,
    )
    from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti

    env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos",
                      render_mode="sensors", sim_backend="gpu",
                      reward_mode="normalized_dense")
    envs = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)
    base = envs.unwrapped
    for wrapper_class, kw in [
        (StateOnlyTensorToDictWrapper, {}),
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
        (DebugRewardWrapper, {}),
    ]:
        envs = wrapper_class(envs, **kw)
    envs = FlattenRGBDObservationWrapperMulti(
        envs, rgb=True, depth=False, state=False, oracle=False, joints=True,
        target_cameras=("base_camera", "hand_camera"))
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True,
                              record_metrics=True)
    return envs, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(ENTRY), required=True)
    ap.add_argument("--env-id", default="RememberColor5-v0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-ball", action="store_true",
                    help="Intercept tasks: log ball pos/vel from the base env")
    args_cli = ap.parse_args()

    os.makedirs(args_cli.out, exist_ok=True)
    device = torch.device("cuda")

    M = importlib.import_module(ENTRY[args_cli.method])
    M.args = M.Args(env_id=args_cli.env_id, include_rgb=True, include_joints=True,
                    capture_video=False, save_model=False,
                    **EXTRA_ARGS.get(args_cli.method, {}))

    base_env = None
    if args_cli.record_ball:
        envs, base_env = build_intercept_envs(args_cli.env_id, args_cli.num_envs)
    else:
        envs = build_eval_envs(M, args_cli.env_id, args_cli.num_envs, args_cli.seed)
    obs, info = envs.reset(seed=args_cli.seed)

    agent = M.Agent(envs, sample_obs=obs).to(device)
    state = torch.load(args_cli.ckpt, map_location=device)
    agent.load_state_dict(state)
    agent.eval()

    # capture the encoder output via forward hook (avoids a second DINOv2 pass)
    emb_slot = {}
    agent.feature_net.register_forward_hook(
        lambda mod, inp, out: emb_slot.__setitem__("v", out.detach()))
    # for FFM also capture the post-mix readout y (what the policy sees);
    # for GRU/LSTM the readout IS the hidden state, so no extra capture needed
    y_slot = {}
    if args_cli.method == "ffm":
        agent.gru.register_forward_hook(
            lambda mod, inp, out: y_slot.__setitem__("v", out[0].detach()))

    H = agent.lstm.hidden_size if hasattr(agent, "lstm") else agent.gru.hidden_size
    is_lstm = args_cli.method == "lstm"
    N, T = args_cli.num_envs, args_cli.horizon

    for b in range(args_cli.batches):
        obs, info = envs.reset(seed=args_cli.seed + 1000 * (b + 1))
        color = (info["oracle_info"].detach().cpu().numpy().astype(np.int64)
                 if not args_cli.record_ball else np.zeros(N, dtype=np.int64))
        if is_lstm:
            rnn_state = (torch.zeros(1, N, H, device=device),
                         torch.zeros(1, N, H, device=device))
        else:
            rnn_state = torch.zeros(1, N, H, device=device)
        done = torch.zeros(N, device=device)

        h_log = np.zeros((T, N, H), dtype=np.float16)
        c_log = np.zeros((T, N, H), dtype=np.float16) if is_lstm else None
        emb_log = None
        y_log = (np.zeros((T, N, agent.gru.output_size), dtype=np.float16)
                 if args_cli.method == "ffm" else None)
        succ_log = np.zeros((T, N), dtype=np.bool_)
        ball_pos_log = np.zeros((T, N, 3), dtype=np.float32) if base_env else None
        ball_vel_log = np.zeros((T, N, 3), dtype=np.float32) if base_env else None

        for t in range(T):
            with torch.no_grad():
                action, rnn_state = agent.get_action(obs, rnn_state, done,
                                                     deterministic=True)
            emb = emb_slot["v"]
            if emb_log is None:
                emb_log = np.zeros((T, N, emb.shape[-1]), dtype=np.float16)
            emb_log[t] = emb.float().cpu().numpy()
            h_log[t] = (rnn_state[0] if is_lstm else rnn_state)[0].float().cpu().numpy()
            if is_lstm:
                c_log[t] = rnn_state[1][0].float().cpu().numpy()
            if y_log is not None:
                y_log[t] = y_slot["v"].reshape(N, -1).float().cpu().numpy()
            if base_env is not None:
                ball_pos_log[t] = base_env.ball.pose.p.cpu().numpy()
                ball_vel_log[t] = base_env.ball.linear_velocity.cpu().numpy()
            obs, rew, term, trunc, info = envs.step(action)
            succ_log[t] = info["success"].detach().cpu().numpy()
            done = (term | trunc).float()

        out = dict(h=h_log, emb=emb_log, succ=succ_log, color=color)
        if is_lstm:
            out["c"] = c_log
        if y_log is not None:
            out["y"] = y_log
        if base_env is not None:
            out["ball_pos"] = ball_pos_log
            out["ball_vel"] = ball_vel_log
        np.savez_compressed(os.path.join(args_cli.out, f"batch{b:03d}.npz"), **out)
        print(f"batch {b + 1}/{args_cli.batches}  "
              f"succ_once={succ_log.any(0).mean():.2f}  "
              f"succ_end={succ_log[-1].mean():.2f}", flush=True)

    envs.close()


if __name__ == "__main__":
    main()
