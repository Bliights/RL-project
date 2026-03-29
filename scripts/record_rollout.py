from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import highway_env  # noqa: F401

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def record_rollouts(
    model_path: Path,
    video_dir: Path,
    n_episodes: int = 3,
    seed: int = 0,
) -> None:
    """Record n_episodes of a trained model and save videos to video_dir."""
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gymnasium.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda _: True,
        name_prefix=f"dqn_{model_path.stem}_seed{seed}",
    )

    model = DQN.load(str(model_path))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"Episode {ep + 1}/{n_episodes} — reward: {total_reward:.2f}")

    env.close()
    print(f"\nVideos saved to {video_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record rollout videos of a trained DQN")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the .zip model (e.g. results/sb3_dqn/seed_0/best_model/best_model.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Video output directory (default: videos/<model_stem>/seed<seed>/)",
    )
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_dir = args.output_dir or Path("videos") / args.model_path.stem / f"seed{args.seed}"
    record_rollouts(
        model_path=args.model_path,
        video_dir=video_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
