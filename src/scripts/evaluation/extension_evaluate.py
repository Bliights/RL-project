import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from rl_project.benchmark.benchmark import HighwayBenchmark
from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.models.dqn.model import DQNModel
from scripts.evaluation.config import DEFAULT_OUTPUT_DIR
from scripts.utils.benchmark_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from scripts.utils.cache import CacheManager
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

EXTENSION_CONFIGS = {
    "shared": (SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG),
}


@app.command()
def main(
    extension: Annotated[str, typer.Option("--extension", "-e")],
    model_path: Annotated[
        Path,
        typer.Option("--model-path", "-m", exists=True, file_okay=True, dir_okay=False),
    ],
    seed: Annotated[int, typer.Option("--seed", "-s")],
    n_episodes: Annotated[int, typer.Option("--n-episodes", "-n", min=1)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
) -> None:
    setup_logging()
    logger.info(f"Loading model from {model_path}")
    model = DQNModel.load(model_path)

    env_id, env_config = EXTENSION_CONFIGS[extension]
    benchmark = HighwayBenchmark(
        config=BenchmarkConfig(
            env_id=env_id,
            env_config=env_config,
        ),
    )

    episodes_data = []

    for ep in range(n_episodes):
        env = benchmark.make_env(seed=seed + ep)
        obs, _ = env.reset()

        done = False
        total_reward = 0.0
        length = 0
        crashed = False
        offroad = False
        speeds = []
        lane_changes = 0
        prev_lane = None

        while not done:
            obs_flat = obs.flatten().astype(np.float32)
            action = model.act(obs_flat)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            length += 1
            speeds.append(info.get("speed", 0.0))

            # tracker changement de voie
            current_lane = info.get("lane_index", (0, 0, 0))[2]
            if prev_lane is not None and current_lane != prev_lane:
                lane_changes += 1
            prev_lane = current_lane

            if terminated:
                crashed = info.get("crashed", False)
                offroad = info.get("offroad", False)

            done = terminated or truncated

        env.close()

        episodes_data.append(
            {
                "episode": ep,
                "seed": seed + ep,
                "reward": total_reward,
                "length": length,
                "crashed": crashed,
                "offroad": offroad,
                "mean_speed": float(np.mean(speeds)),
                "lane_changes": lane_changes,  # ← nouvelle métrique
            }
        )

        logger.info(
            f"Episode {ep}: reward={total_reward:.2f} crashed={crashed} lane_changes={lane_changes}"
        )

    # Sauvegarde
    eval_dir = output_dir / "extension" / extension / f"seed_{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(episodes_data)
    history_path = eval_dir / f"evaluation_history_dqn_seed_{seed}.csv"
    CacheManager.save(history_df, history_path)

    summary = {
        "mean_reward": history_df["reward"].mean(),
        "std_reward": history_df["reward"].std(),
        "crash_rate": history_df["crashed"].mean(),
        "offroad_rate": history_df["offroad"].mean(),
        "mean_speed": history_df["mean_speed"].mean(),
        "mean_lane_changes": history_df["lane_changes"].mean(),  # ← nouvelle métrique
        "n_episodes": n_episodes,
    }
    summary_path = eval_dir / "evaluation_summary.json"
    CacheManager.save(summary, summary_path)
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    app()
