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
from rl_project.models.factory import build_model, load_model
from scripts.training.config import (
    DEFAULT_CHECKPOINT_EVERY_EPISODES,
    DEFAULT_DQN_CONFIG,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_EVERY_EPISODES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SB3_CONFIG,
)
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


@app.command(help="")
def main(
    model: Annotated[
        ModelType,
        typer.Option("--model", "-m", help="Type of model to use for the training"),
    ],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Base seed to use for the training")],
    n_steps: Annotated[
        int,
        typer.Option("--n-steps", "-n", min=1, help="Number of training steps to do"),
    ],
    checkpoint_every_episodes: Annotated[
        int,
        typer.Option(
            "--checkpoint-every-episodes",
            "-c",
            help="Interval of training episode between each checkpoints",
        ),
    ] = DEFAULT_CHECKPOINT_EVERY_EPISODES,
    eval_every_episodes: Annotated[
        int,
        typer.Option(
            "--eval-every-episodes",
            help="Interval of training episode between each evaluation",
        ),
    ] = DEFAULT_EVAL_EVERY_EPISODES,
    eval_episodes: Annotated[
        int,
        typer.Option(
            "--eval-episodes",
            help="Number of episodes for each evaluation",
        ),
    ] = DEFAULT_EVAL_EPISODES,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Path to the output directory for the modeles, checkpoints and metrics",
        ),
    ] = DEFAULT_OUTPUT_DIR,
    resume_from: Annotated[
        Path | None,
        typer.Option(
            "--resume-from",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Path to a saved model to resume from",
        ),
    ] = None,
) -> None:
    """
    Train a RL model on the benchmark environment

    Parameters
    ----------
    model : ModelType
        Type of RL model to train
    seed : int
        Random seed used for reproducibility
    n_steps : int
        Number of training steps to run
    checkpoint_every_episodes : int
        Interval of training episode between each checkpoints
    eval_every_episodes : int
        Interval of training episode between each evaluation
    eval_episodes : int
        Number of episodes for each evaluation
    output_dir : Annotated[Path, typer.Option, optional
        Directory where training outputs are saved
    resume_from : Path  |  None
        Path to a saved model to resume from
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

    if resume_from is not None:
        logger.info(f"Loading model from {resume_from}")
        rl_model = load_model(
            model_type=model,
            model_path=resume_from,
        )
        logger.info("Loaded existing model !")
    else:
        if model == ModelType.DQN:
            config = DEFAULT_DQN_CONFIG
        elif model == ModelType.SB3:
            config = DEFAULT_SB3_CONFIG
        else:
            raise ValueError(f"Unsupported model type: {model}")

        rl_model = build_model(
            model_type=model,
            obs_dim=obs_dim,
            n_actions=n_actions,
            config=config,
        )
        logger.info("New model created !")

    train_dir = output_dir / model.value / f"seed_{seed}"
    train_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Start of training for {n_steps} steps...")

    history = rl_model.fit(
        env=env,
        output_dir=train_dir,
        seed=seed,
        n_steps=n_steps,
        checkpoint_every_episodes=checkpoint_every_episodes,
        eval_every_episodes=eval_every_episodes,
        eval_episodes=eval_episodes,
    )

    logger.info("Training finished !")

    final_model_path = train_dir / f"model_{model.value}_seed_{seed}_final.pt"
    rl_model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    history_df = pd.DataFrame([step.to_dict() for step in history])
    history_path = train_dir / f"training_history_{model.value}_seed_{seed}.csv"
    CacheManager.save(history_df, history_path)
    logger.info(
        f"Training history saved to {history_path}",
    )

    env.close()


if __name__ == "__main__":
    app()
