from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import EVAL_SEEDS, make_env


def evaluate_model(
    model_path: Path,
    n_episodes: int = 50,
    seed: int = 0,
) -> dict[str, float]:
    """Run deterministic evaluation and return aggregated metrics."""
    model = DQN.load(str(model_path))
    env = make_env(seed=seed)()

    rewards: list[float] = []
    lengths: list[int] = []
    crashes: list[float] = []
    offroads: list[float] = []
    ep_speeds: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        length = 0
        step_speeds: list[float] = []
        crashed = False
        offroad = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            step_speeds.append(float(info.get("speed", 0.0)))
            if terminated or truncated:
                done = True
                crashed = bool(info.get("crashed", False))
                offroad = not bool(info.get("on_road", True))

        rewards.append(total_reward)
        lengths.append(length)
        crashes.append(float(crashed))
        offroads.append(float(offroad))
        ep_speeds.append(float(np.mean(step_speeds)) if step_speeds else 0.0)

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "offroad_rate": float(np.mean(offroads)),
        "mean_speed": float(np.mean(ep_speeds)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SB3 DQN model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the .zip model file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save eval_results.json (defaults to model parent dir)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_eval_episodes,
        seed=args.seed,
    )

    # infer seed from parent dir name (seed_0, seed_1, ...)
    seed_name = args.model_path.parent.name
    if seed_name.startswith("seed_"):
        results["seed"] = int(seed_name.split("_")[1])

    results["model_path"] = str(args.model_path)

    output_dir = args.output_dir or args.model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eval_results.json"

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results ({args.n_eval_episodes} episodes):")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSaved to {out_path}")
