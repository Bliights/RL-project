import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rl_project.benchmark.benchmark import HighwayBenchmark
from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.models.core.typing import ModelType
from rl_project.models.dqn.model import DQNModel
from scripts.evaluation.config import DEFAULT_OUTPUT_DIR
from scripts.utils.benchmark_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
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

    if model_type == ModelType.DQN:
        model = DQNModel.load(model_path)

    benchmark = HighwayBenchmark(
        config=BenchmarkConfig(
            env_id=SHARED_CORE_ENV_ID,
            env_config=SHARED_CORE_CONFIG,
        ),
    )
    logger.info("Benchmark loaded !")
    logger.info("Starting evaluation...")

    summary, episodes = model.evaluate(
        env_factory=benchmark.make_env,
        n_episodes=n_episodes,
        seed=seed,
    )

    logger.info(
        f"Evaluation summary: {summary.to_dict()}",
    )

    eval_dir = output_dir / "extension" / extension / model_type.value / f"seed_{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "evaluation_summary.json"
    CacheManager.save(summary.to_dict(), summary_path)
    logger.info(f"Saved summary to {summary_path}")

    history_df = pd.DataFrame([ep.to_dict() for ep in episodes])
    history_path = eval_dir / f"evaluation_history_{model_type.value}_seed_{seed}.csv"
    CacheManager.save(history_df, history_path)
    logger.info(f"Evaluation history saved to {history_path}")


if __name__ == "__main__":
    app()
