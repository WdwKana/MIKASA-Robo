import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a few rendered frames from a MIKASA-Robo task."
    )
    parser.add_argument("--env-id", default="RememberColor9-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-env-idx", type=int, default=0)
    parser.add_argument(
        "--save-steps",
        nargs="+",
        type=int,
        default=[0, 5, 10, 20, 30, 40, 50],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <script_dir>/<env_id>_frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Importing dependencies...", flush=True)

    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import torch
    import gymnasium as gym
    import mikasa_robo_suite
    from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper

    _ = mikasa_robo_suite  # Registers MIKASA-Robo gym environments.

    def resolve_output_dir():
        script_dir = Path(__file__).resolve().parent
        if args.output_dir is None:
            output_dir = script_dir / f"{args.env_id}_frames"
        else:
            candidate = Path(args.output_dir).expanduser()
            output_dir = candidate if candidate.is_absolute() else script_dir / candidate
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_max_episode_steps():
        spec = gym.spec(args.env_id)
        if spec is None or spec.max_episode_steps is None:
            raise ValueError(f"Could not determine max_episode_steps for {args.env_id}.")
        return int(spec.max_episode_steps)

    def to_torch_action(action):
        if isinstance(action, dict):
            return {key: to_torch_action(value) for key, value in action.items()}
        if torch.is_tensor(action):
            return action
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action)
        return torch.as_tensor(action)

    def render_batch(env):
        for method_name in ("render", "render_all", "render_rgb_array"):
            method = getattr(env, method_name, None)
            if callable(method):
                frame_batch = method()
                if frame_batch is not None:
                    return frame_batch, method_name
        raise RuntimeError(
            "No usable render method found. Tried render(), render_all(), and render_rgb_array()."
        )

    def batched_frame_to_image(frame_batch):
        if torch.is_tensor(frame_batch):
            frame_batch = frame_batch.detach().cpu().numpy()
        elif isinstance(frame_batch, (list, tuple)):
            frame_batch = np.stack(frame_batch)
        else:
            frame_batch = np.asarray(frame_batch)

        if frame_batch.ndim == 5:
            frame = frame_batch[args.save_env_idx, 0]
        elif frame_batch.ndim == 4:
            frame = frame_batch[args.save_env_idx]
        elif frame_batch.ndim == 3:
            frame = frame_batch
        else:
            raise ValueError(f"Unexpected render output shape: {frame_batch.shape}")

        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        if np.issubdtype(frame.dtype, np.floating):
            if np.max(frame) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255)
        else:
            frame = np.clip(frame, 0, 255)

        return frame.astype(np.uint8)

    def save_frame(frame_batch, step, output_dir, prefix):
        frame = batched_frame_to_image(frame_batch)
        output_path = output_dir / f"{prefix}_env{args.save_env_idx}_step{step:03d}.png"
        Image.fromarray(frame).save(output_path)
        print(f"Saved step {step:03d} -> {output_path}", flush=True)

    def all_envs_done(terminated, truncated):
        if torch.is_tensor(terminated):
            return bool(torch.all(terminated | truncated).item())
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        return bool(np.all(np.logical_or(terminated, truncated)))

    save_steps = sorted(set(args.save_steps))
    if not save_steps:
        raise ValueError("save_steps cannot be empty.")
    if any(step < 0 for step in save_steps):
        raise ValueError("save_steps must be non-negative.")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if not 0 <= args.save_env_idx < args.num_envs:
        raise ValueError("save_env_idx must be in [0, num_envs - 1].")

    output_dir = resolve_output_dir()
    max_episode_steps = get_max_episode_steps()
    if max(save_steps) > max_episode_steps:
        print(
            f"Requested steps go beyond max_episode_steps={max_episode_steps}; "
            "steps outside the episode horizon will be skipped.",
            flush=True,
        )
        save_steps = [step for step in save_steps if step <= max_episode_steps]
        if not save_steps:
            raise ValueError("All requested steps are outside the episode horizon.")

    last_step = max(save_steps)

    print(
        f"Creating env {args.env_id} "
        f"(num_envs={args.num_envs}, sim_backend={args.sim_backend})...",
        flush=True,
    )
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb",
        render_mode="all",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
    )
    env = StateOnlyTensorToDictWrapper(env)

    saved_steps = []
    try:
        print(f"Output dir: {output_dir}", flush=True)
        print(f"Capturing steps: {save_steps}", flush=True)
        print("Resetting env...", flush=True)
        obs, _ = env.reset(seed=args.seed)
        print(f"Observation keys: {list(obs.keys())}", flush=True)

        if 0 in save_steps:
            frame_batch, render_method = render_batch(env)
            save_frame(frame_batch, step=0, output_dir=output_dir, prefix=render_method)
            saved_steps.append(0)

        for step in tqdm(range(1, last_step + 1), desc="Stepping env", dynamic_ncols=True):
            action = to_torch_action(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)

            if step in save_steps:
                frame_batch, render_method = render_batch(env)
                save_frame(frame_batch, step=step, output_dir=output_dir, prefix=render_method)
                saved_steps.append(step)

            if all_envs_done(terminated, truncated):
                if step < last_step:
                    print(
                        f"All environments finished at step {step}; stopping early.",
                        flush=True,
                    )
                break
    finally:
        env.close()

    print(f"Done. Saved {len(saved_steps)} frame(s) to {output_dir}", flush=True)


if __name__ == "__main__":
    main()