from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

import highway_env  # noqa: F401, E402

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID  # noqa: E402

# Config for recording: longer episodes, denser traffic, 60 Hz for smooth video
RECORD_CONFIG = {
    **SHARED_CORE_CONFIG,
    "duration": 60,
    "vehicles_count": 50,
    "simulation_frequency": 60,  # 60 Hz — fluid video
    "policy_frequency": 2,  # 2 decisions/s — keeps action pace reasonable at 60 Hz
}


def run_episode(model: DQN, seed: int) -> dict:
    """Run one episode without recording, collect stats."""
    env = gymnasium.make(SHARED_CORE_ENV_ID, config=RECORD_CONFIG, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    length = 0
    crashed = False
    lane_changes = 0
    prev_lane = env.unwrapped.vehicle.lane_index[2]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        length += 1
        current_lane = env.unwrapped.vehicle.lane_index[2]
        if current_lane != prev_lane:
            lane_changes += 1
            prev_lane = current_lane
        if terminated or truncated:
            crashed = bool(info.get("crashed", False))
            done = True

    env.close()
    return {
        "seed": seed,
        "reward": total_reward,
        "length": length,
        "crashed": crashed,
        "lane_changes": lane_changes,
    }


def record_episode(model: DQN, seed: int, video_dir: Path, label: str) -> None:
    env = gymnasium.make(SHARED_CORE_ENV_ID, config=RECORD_CONFIG, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda _: True,
        name_prefix=label,
    )
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


def record_rollouts(
    model_path: Path,
    video_dir: Path,
    n_crashes: int = 3,
    n_low_reward: int = 3,
    n_active: int = 3,
    n_candidates: int = 60,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    model = DQN.load(str(model_path))

    print(f"Scanning {n_candidates} seeds...")
    stats = []
    for seed in range(n_candidates):
        s = run_episode(model, seed)
        stats.append(s)
        print(
            f"  seed {seed:3d} — reward: {s['reward']:6.2f}  length: {s['length']:3d}"
            f"  lane_changes: {s['lane_changes']:2d}  crashed: {s['crashed']}",
        )

    crashes = [s for s in stats if s["crashed"]]
    survived = [s for s in stats if not s["crashed"]]
    low_reward = sorted(survived, key=lambda s: s["reward"])
    active = sorted(survived, key=lambda s: -s["lane_changes"])

    to_record: list[tuple[str, dict]] = []
    for i, s in enumerate(crashes[:n_crashes]):
        to_record.append((f"crash_{i + 1:02d}_seed{s['seed']}_r{s['reward']:.0f}", s))
    for i, s in enumerate(low_reward[:n_low_reward]):
        to_record.append((f"lowreward_{i + 1:02d}_seed{s['seed']}_r{s['reward']:.0f}", s))
    for i, s in enumerate(active[:n_active]):
        to_record.append((f"active_{i + 1:02d}_seed{s['seed']}_lc{s['lane_changes']}", s))

    print(f"\nRecording {len(to_record)} episodes...")
    for label, s in to_record:
        print(f"  {label}")
        record_episode(model, s["seed"], video_dir, label)

    print(f"\nVideos saved to {video_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record interesting rollout videos of a trained DQN",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-crashes", type=int, default=3)
    parser.add_argument("--n-low-reward", type=int, default=3)
    parser.add_argument("--n-active", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_dir = args.output_dir or Path("videos") / "curated"
    record_rollouts(
        model_path=args.model_path,
        video_dir=video_dir,
        n_crashes=args.n_crashes,
        n_low_reward=args.n_low_reward,
        n_active=args.n_active,
        n_candidates=args.n_candidates,
    )
