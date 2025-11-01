from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def collect_success_trajectories(src_dir: Path) -> tuple[np.ndarray, ...]:
    npz_paths = sorted(
        src_dir.glob("train_data_*.npz"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    states, actions, rewards, terminals, images, traj_lengths = [], [], [], [], [], []

    for path in npz_paths:
        with np.load(path) as episode:
            if episode["success"].max() == 0:
                continue

            done = episode["done"]
            dones = np.where(done > 0)[0]
            horizon = int(dones[0]) + 1 if dones.size else len(done)

            states.append(episode["joints"][:horizon].astype(np.float32))
            actions.append(episode["action"][:horizon].astype(np.float32))
            rewards.append(episode["reward"][:horizon].astype(np.float32))
            terminals.append(done[:horizon].astype(np.float32))

            rgb = episode["rgb"][:horizon, :, :, :3]
            images.append(np.transpose(rgb, (0, 3, 1, 2)).astype(np.uint8))

            traj_lengths.append(horizon)

    if not traj_lengths:
        raise RuntimeError(f"No successful trajectories found in {src_dir}")

    states_arr = np.concatenate(states)
    actions_arr = np.concatenate(actions)
    rewards_arr = np.concatenate(rewards)
    terminals_arr = np.concatenate(terminals)
    images_arr = np.concatenate(images)
    traj_lengths_arr = np.asarray(traj_lengths, dtype=np.int64)

    return states_arr, actions_arr, rewards_arr, terminals_arr, images_arr, traj_lengths_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge successful trajectories into a single NPZ file.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/MIKASA-Robo/unbatched/RememberShapeAndColor3x2-v0"),
        help="Directory containing unbatched *.npz trajectories.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/mikasa/RememberShapeAndColor3x2-v0/RememberShapeAndColor3x2-v0_from_unbatched.npz"),
        help="Output file path for the merged dataset.",
    )
    args = parser.parse_args()

    states, actions, rewards, terminals, images, traj_lengths = collect_success_trajectories(args.src)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.dst,
        states=states,
        actions=actions,
        traj_lengths=traj_lengths,
        rewards=rewards,
        terminals=terminals,
        images=images,
    )

    print(f"Saved {traj_lengths.size} trajectories to {args.dst}")


if __name__ == "__main__":
    main()

