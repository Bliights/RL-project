import logging
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer

from rl_project.benchmark.benchmark import HighwayBenchmark
from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.models.core.typing import ModelType
from rl_project.models.factory import build_model
from scripts.training.config import DEFAULT_DQN_CONFIG, DEFAULT_OUTPUT_DIR
from scripts.utils.benchmark_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from scripts.utils.cache import CacheManager
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for reproducibility

    Parameters
    ----------
    seed : int
        Random seed used to make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    model: Annotated[ModelType, typer.Option("--model", "-m")],
    seed: Annotated[int, typer.Option("--seed", "-s")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Train a RL model on the benchmark environment

    Parameters
    ----------
    model : ModelType
        Type of RL model to train
    seed : int
        Random seed used for reproducibility
    output_dir : Annotated[Path, typer.Option, optional
        Directory where training outputs are saved
    """
    setup_logging()
    set_global_seed(seed)

    logger.info(f"Starting training with model={model.value} seed={seed}")

    benchmark = HighwayBenchmark(
        config=BenchmarkConfig(
            env_id=SHARED_CORE_ENV_ID,
            env_config=SHARED_CORE_CONFIG,
        ),
    )
    env = benchmark.make_env(seed=seed)
    logger.info("Benchmark loaded !")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if model == ModelType.DQN:
        config = DEFAULT_DQN_CONFIG
    elif model == ModelType.SB3:
        config = None

    rl_model = build_model(
        model_type=model,
        obs_dim=obs_dim,
        n_actions=n_actions,
        config=config,
    )

    train_dir = output_dir / model.value / f"seed_{seed}"
    train_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model created !")
    logger.info("Strat of the training...")

    history = rl_model.fit(
        env=env,
        output_dir=train_dir,
        seed=seed,
    )

    logger.info(f"Training finished: {len(history)} episodes")

    final_model_path = train_dir / f"model_{model.value}_seed_{seed}.pt"
    rl_model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    history_df = pd.DataFrame([episode.to_dict() for episode in history])
    history_path = train_dir / f"training_history_{model.value}_seed_{seed}.csv"
    CacheManager.save(history_df, history_path)
    logger.info(
        f"Training history saved to {history_path}",
    )

    env.close()


if __name__ == "__main__":
    app()
