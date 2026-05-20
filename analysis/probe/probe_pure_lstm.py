"""
Probe pure-LSTM (NatureCNN + LSTM) on IMed, parallel to probe_state_for_ball.py
and probe_action_and_contact.py used for V1/V2/V3a.

Pure-LSTM is the simplest possible LSTM+PPO baseline: it has no DINOv2, no
saliency head, no K-V buffer, no fuse layer. The "s_t" fed to the actor is
the LSTM hidden output directly. If this baseline shows the same
mechanistic signatures we observed in V3a-LSTM-Hyb (e.g., high SR_end on
IMed, similar action / state structure), that is cross-method evidence that
LSTM-the-component is what carries the inductive bias for motion tasks,
independent of perception / memory architecture.

Outputs per checkpoint:
  - ball_position_{linear, mlp}: R² of s_t (= lstm_hidden) -> ball position
  - ball_velocity_{linear, mlp}: same -> linear velocity
  - tcp_to_ball_{linear, mlp}: same -> gripper↔ball vector
  - action_smoothness: ‖a_t - a_{t-1}‖² split by distance bin, succ/fail
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
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs import *  # noqa: F401
from mikasa_robo_suite.utils.wrappers import *
from baselines.ppo.utils_ebm import FlattenRGBDObservationWrapperMulti

DEVICE = torch.device("cuda")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Replicates the NatureCNN from ppo_memtasks_lstm_dual.py."""
    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256
        self.list_of_obs_keys = list(sample_obs.keys())

        if 'rgb' in self.list_of_obs_keys:
            in_channels = sample_obs["rgb"].shape[-1]
            cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flat, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        for key in self.list_of_obs_keys:
            if key in ['oracle_info', 'prompt']:
                extractors[key] = nn.Sequential(nn.Linear(sample_obs[key].shape[-1], 64), nn.ReLU())
                self.out_features += 64
            elif key == 'joints':
                extractors[key] = nn.Sequential(nn.Linear(sample_obs[key].shape[-1], 256), nn.ReLU())
                self.out_features += 256

        if 'state' in sample_obs.keys():
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, obs):
        feats = []
        for k, m in self.extractors.items():
            x = obs[k]
            if k == "rgb":
                x = x.float().permute(0,3,1,2) / 255.0
            elif k in ('oracle_info', 'prompt', 'joints'):
                x = x.float()
            feats.append(m(x))
        return torch.cat(feats, dim=1)


class PureLSTMAgent(nn.Module):
    def __init__(self, sample_obs, action_dim, lstm_hidden=512, lstm_layers=1):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs)
        latent = self.feature_net.out_features
        self.lstm_hidden_size = lstm_hidden
        self.lstm = nn.LSTM(latent, lstm_hidden, lstm_layers, dropout=0.0, batch_first=False)
        for n, p in self.lstm.named_parameters():
            if "bias" in n: nn.init.constant_(p, 0.0)
            elif "weight" in n: nn.init.orthogonal_(p, 1.0)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01*np.sqrt(2)),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)


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


@torch.no_grad()
def collect_rollouts(agent, env, num_episodes, num_envs, lstm_hidden_size):
    obs, _ = env.reset(seed=12345)
    h = torch.zeros(1, num_envs, lstm_hidden_size, device=DEVICE)
    c = torch.zeros(1, num_envs, lstm_hidden_size, device=DEVICE)
    pending = [[] for _ in range(num_envs)]
    out = {"s":[], "a":[], "t2b":[], "ball":[], "vel":[], "step":[], "ep":[], "succ":[]}
    ep_id = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    step_in_ep = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    done_prev = torch.zeros(num_envs, device=DEVICE)
    eps_done = 0

    while eps_done < num_episodes:
        # reset h,c for done envs (as in training)
        d_ = done_prev.view(1, -1, 1).float()
        h = (1.0 - d_) * h
        c = (1.0 - d_) * c

        feats = agent.feature_net(obs).unsqueeze(0)
        h_out, (h, c) = agent.lstm(feats, (h, c))
        s_t = h_out.squeeze(0)  # (B, lstm_hidden)

        inner = env._env.unwrapped
        ball_p = inner.ball.pose.p
        tcp_p = inner.agent.tcp.pose.p
        ball_v = inner.ball.linear_velocity
        t2b = ball_p - tcp_p

        act = agent.actor_mean(s_t)
        s_np = s_t.detach().cpu().numpy(); a_np = act.detach().cpu().numpy()
        ball_np = ball_p.detach().cpu().numpy()
        vel_np = ball_v.detach().cpu().numpy()
        t2b_np = t2b.detach().cpu().numpy()
        step_np = step_in_ep.cpu().numpy(); ep_np = ep_id.cpu().numpy()

        for e in range(num_envs):
            pending[e].append((s_np[e], a_np[e], t2b_np[e], ball_np[e], vel_np[e],
                              int(step_np[e]), int(ep_np[e])))

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
                    out["t2b"].append(rec[2]); out["ball"].append(rec[3])
                    out["vel"].append(rec[4])
                    out["step"].append(rec[5]); out["ep"].append(rec[6])
                    out["succ"].append(is_succ)
                pending[e] = []; ep_id[e] += 1; eps_done += 1
        done_prev = done.float()
        step_in_ep = torch.where(done, torch.zeros_like(step_in_ep), step_in_ep + 1)
    for k in out: out[k] = np.array(out[k])
    return out


def fit_linear(X, Y, ridge=1e-2):
    N = X.shape[0]
    rng = np.random.default_rng(0); perm = rng.permutation(N); s = int(0.8*N)
    tr, te = perm[:s], perm[s:]
    Xtr = np.concatenate([X[tr], np.ones((s,1))], axis=1)
    Xte = np.concatenate([X[te], np.ones((N-s,1))], axis=1)
    D = Xtr.shape[1]
    A = Xtr.T @ Xtr + ridge*np.eye(D); A[-1,-1] -= ridge
    W = np.linalg.solve(A, Xtr.T @ Y[tr])
    Yhat = Xte @ W; Yte = Y[te]
    ss_res = ((Yte-Yhat)**2).sum(axis=0); ss_tot = ((Yte-Yte.mean(0))**2).sum(axis=0)
    return {"r2_per_axis": (1 - ss_res/np.maximum(ss_tot,1e-8)).tolist(),
            "r2_overall": float(1 - ss_res.sum()/max(ss_tot.sum(),1e-8))}


def fit_mlp(X, Y, hidden=256, epochs=80, lr=1e-3, batch=1024):
    N, D = X.shape
    if Y.ndim == 1: Y = Y[:, None]
    K = Y.shape[1]
    rng = np.random.default_rng(0); perm = rng.permutation(N); s = int(0.8*N)
    tr, te = perm[:s], perm[s:]
    Xtr = torch.from_numpy(X[tr]).float().to(DEVICE); Ytr = torch.from_numpy(Y[tr]).float().to(DEVICE)
    Xte = torch.from_numpy(X[te]).float().to(DEVICE); Yte = torch.from_numpy(Y[te]).float().to(DEVICE)
    m = nn.Sequential(nn.Linear(D, hidden), nn.GELU(),
                      nn.Linear(hidden, hidden), nn.GELU(),
                      nn.Linear(hidden, K)).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    n_tr = Xtr.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, batch):
            b = idx[i:i+batch]
            loss = ((m(Xtr[b]) - Ytr[b])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): Yhat = m(Xte).cpu().numpy()
    Yte_np = Yte.cpu().numpy()
    ss_res = ((Yte_np-Yhat)**2).sum(axis=0); ss_tot = ((Yte_np-Yte_np.mean(0))**2).sum(axis=0)
    return {"r2_per_axis": (1 - ss_res/np.maximum(ss_tot,1e-8)).tolist(),
            "r2_overall": float(1 - ss_res.sum()/max(ss_tot.sum(),1e-8))}


def action_smoothness(data, near_thresh=0.10):
    a = data["a"]; ep = data["ep"]; succ = data["succ"]
    dist = np.linalg.norm(data["t2b"], axis=1)
    d = np.full(a.shape[0], np.nan)
    for i in range(1, a.shape[0]):
        if ep[i] != ep[i-1]: continue
        d[i] = np.sum((a[i] - a[i-1])**2)
    valid = ~np.isnan(d)
    def stat(m):
        x = d[m & valid]
        return {"mean": float(x.mean()) if x.size else float("nan"),
                "n": int(x.size)}
    near = dist < near_thresh
    sm = succ > 0.5
    return {
        "overall": stat(np.ones_like(valid, dtype=bool)),
        "near_ball": stat(near), "far_from_ball": stat(~near),
        "success_eps": stat(sm), "failure_eps": stat(~sm),
        "near_in_success": stat(near & sm),
        "near_in_failure": stat(near & ~sm),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="InterceptMedium-v0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episodes", type=int, default=32)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    env = build_env(args.env_id, args.num_envs)
    obs, _ = env.reset(seed=0)
    act_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    agent = PureLSTMAgent(obs, act_dim, lstm_hidden=512).to(DEVICE)
    sd = torch.load(args.checkpoint, map_location=DEVICE)
    # Filter out shape-mismatched buffers
    own = {k: v.shape for k, v in agent.state_dict().items()}
    filt = {k: v for k, v in sd.items() if k not in own or v.shape == own[k]}
    agent.load_state_dict(filt, strict=False)
    agent.eval()
    print(f"[probe-lstm] loaded {args.checkpoint}")

    print(f"[probe-lstm] collecting {args.num_episodes} episodes...")
    data = collect_rollouts(agent, env, args.num_episodes, args.num_envs, agent.lstm_hidden_size)
    print(f"  N={data['s'].shape[0]} steps, s_t={data['s'].shape[1]}d, "
          f"a={data['a'].shape[1]}d, succ_rate={(data['succ']>0.5).mean():.3f}")

    print("[probe-lstm] fitting linear + MLP probes (ball pos/vel/tcp)...")
    out = {
        "method": "pure-lstm",
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "num_steps": int(data["s"].shape[0]),
        "success_rate_in_data": float((data["succ"]>0.5).mean()),
        "ball_position":     fit_linear(data["s"], data["ball"]),
        "ball_position_mlp": fit_mlp(data["s"], data["ball"]),
        "ball_velocity":     fit_linear(data["s"], data["vel"]),
        "ball_velocity_mlp": fit_mlp(data["s"], data["vel"]),
        "tcp_to_ball_vec_linear":  fit_linear(data["s"], data["t2b"]),
        "tcp_to_ball_vec_mlp":     fit_mlp(data["s"], data["t2b"]),
        "tcp_to_ball_dist_linear": fit_linear(data["s"], np.linalg.norm(data["t2b"], axis=1, keepdims=True)),
        "tcp_to_ball_dist_mlp":    fit_mlp(data["s"], np.linalg.norm(data["t2b"], axis=1, keepdims=True)),
        "action_smoothness": action_smoothness(data),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoint"}, indent=2, default=float))
    out_path = args.output or str(Path(args.checkpoint).parent / "probe_pure_lstm.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[probe-lstm] saved to {out_path}")


if __name__ == "__main__":
    main()
