import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rl_project.benchmark.benchmark import HighwayBenchmark
from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.models.core.typing import ModelType
from rl_project.models.factory import load_model
from scripts.envs.factory import get_env_config
from scripts.envs.typing import EnvType
from scripts.evaluation.config import DEFAULT_OUTPUT_DIR
from scripts.utils.cache import CacheManager
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def get_model_type(model_path: Path) -> ModelType:
    """
    Infer the model type from a checkpoint file name

    Parameters
    ----------
    model_path : Path
        Path to the saved model

    Returns
    -------
    ModelType
        Model type extracted from the file name
    """
    name = model_path.stem
    parts = name.split("_")
    return ModelType(parts[1])


def get_env_type(model_path: Path) -> EnvType:
    """
    Infer the env type from a checkpoint file name

    Parameters
    ----------
    model_path : Path
        Path to the saved model

    Returns
    -------
    EnvType
        Env type extracted from the file name
    """
    name = model_path.stem
    parts = name.split("_")
    return EnvType(parts[2])


def get_model_seed(model_path: Path) -> int:
    """
    Get the seed of the model from the checkpoint file

    Parameters
    ----------
    model_path : Path
        Path to the saved model

    Returns
    -------
    Int
        Seed used to train the model
    """
    name = model_path.stem
    parts = name.split("_")
    return int(parts[4])


@app.command(help="")
def main(
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Path of the model to use for the evaluation",
        ),
    ],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Base seed to use for the evaluation")],
    n_episodes: Annotated[
        int,
        typer.Option(
            "--n-episodes",
            "-n",
            min=1,
            help="Number of episodes to do for the evaluation",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Path to the output directory for the results",
        ),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Evaluate a saved RL model on the benchmark environment

    Parameters
    ----------
    model_path : Path
        Path to the saved model checkpoint
    seed : int
        Base random seed used for reproducible evaluation
    n_episodes : int
        Number of evaluation episodes to run
    output_dir : Path
        Directory where evaluation outputs are saved
    """
    setup_logging()

    logger.info(f"Loading model from {model_path}")
    model_type = get_model_type(model_path)
    env_type = EnvType.BASELINE
    model = load_model(model_type, model_path)

    env_id, env_config = get_env_config(env_type)

    benchmark = HighwayBenchmark(
        config=BenchmarkConfig(
            env_id=env_id,
            env_config=env_config,
        ),
    )
    env = benchmark.make_env(seed=seed)
    logger.info("Benchmark loaded !")
    logger.info("Starting evaluation...")

    summary, episodes = model.evaluate(
        env=env,
        n_episodes=n_episodes,
        seed=seed,
        verbose=True,
    )

    logger.info(
        f"Evaluation summary: {summary.to_dict()}",
    )

    env_type = get_env_type(model_path)
    seed = get_model_seed(model_path)

    eval_dir = output_dir / model_type.value / env_type.value / f"seed_{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "evaluation_summary.json"
    CacheManager.save(summary.to_dict(), summary_path)
    logger.info(f"Saved summary to {summary_path}")

    history_df = pd.DataFrame([ep.to_dict() for ep in episodes])
    history_path = (
        eval_dir / f"evaluation_history_{model_type.value}_{env_type.value}_seed_{seed}.csv"
    )
    CacheManager.save(history_df, history_path)
    logger.info(f"Evaluation history saved to {history_path}")

    env.close()


if __name__ == "__main__":
    app()
