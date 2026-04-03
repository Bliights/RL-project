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
from scripts.utils.benchmark_config_extension import (
    EXTENSION_ENV_ID,
    EXTENSION_GREEDY_PASSING_CONFIG,
)
from scripts.utils.cache import CacheManager
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

DOUBLE_DQN_CONFIGS = {
    "baseline": (SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG),
    "greedy": (EXTENSION_ENV_ID, EXTENSION_GREEDY_PASSING_CONFIG),
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    config: Annotated[str, typer.Option("--config", "-c")],
    seed: Annotated[int, typer.Option("--seed", "-s")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
) -> None:
    setup_logging()
    set_global_seed(seed)

    if config not in DOUBLE_DQN_CONFIGS:
        raise ValueError(f"Config '{config}' inconnue. Choisis parmi : {list(DOUBLE_DQN_CONFIGS.keys())}")

    env_id, env_config = DOUBLE_DQN_CONFIGS[config]
    logger.info(f"Starting Double DQN training config={config} seed={seed}")

    benchmark = HighwayBenchmark(
        config=BenchmarkConfig(
            env_id=env_id,
            env_config=env_config,
        ),
    )
    env = benchmark.make_env(seed=seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = build_model(
        model_type=ModelType.DOUBLE_DQN,
        obs_dim=obs_dim,
        n_actions=n_actions,
        config=DEFAULT_DQN_CONFIG,
    )

    train_dir = output_dir / "double_dqn" / config / f"seed_{seed}"
    train_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(env=env, output_dir=train_dir, seed=seed)
    logger.info(f"Training finished: {len(history)} episodes")

    final_model_path = train_dir / f"model_double_dqn_seed_{seed}.pt"
    model.save(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    history_df = pd.DataFrame([ep.to_dict() for ep in history])
    history_path = train_dir / f"training_history_double_dqn_seed_{seed}.csv"
    CacheManager.save(history_df, history_path)

    env.close()


if __name__ == "__main__":
    app()