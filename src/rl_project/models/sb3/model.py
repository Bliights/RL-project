from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    TrainingEpisodeMetrics,
)
from rl_project.models.sb3.typing import SB3Config


class SB3Model(BaseRLModel):
    def __init__(self, obs_dim: int, n_actions: int, config: SB3Config) -> None:
        """
        Initialize the SB3 model wrapper

        Parameters
        ----------
        obs_dim : int
            Dimension of the input observation vector
        n_actions : int
            Number of discrete actions available in the environment
        config : SB3Config
            configuration associated with the SB3 model
        """
        super().__init__(obs_dim=obs_dim, n_actions=n_actions)
        self.config = config

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action from the current policy

        Parameters
        ----------
        state : np.ndarray
            Current environment observation
        greedy : bool, optional
            Whether to use deterministic action selection during inference

        Returns
        -------
        int
            Action selected by the policy
        """
        raise NotImplementedError("SB3 model not implemented yet.")

    def fit(self, env: gym.Env, output_dir: Path, seed: int) -> list[TrainingEpisodeMetrics]:
        """
        Train the model on the provided environment

        Parameters
        ----------
        env : gym.Env
            Gymnasium environment used for training
        output_dir : Path
            Directory where training checkpoints should be saved
        seed : int
            Random seed used for reproducibility

        Returns
        -------
        list[TrainingEpisodeMetrics]
            List of training metrics collected at the episode level
        """
        raise NotImplementedError("SB3 model not implemented yet.")

    def evaluate(
        self,
        env_factory: Callable[[int], gym.Env],
        n_episodes: int,
        seed: int,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the policy over several episodes

        Parameters
        ----------
        env_factory : Callable[[int], gym.Env]
            Factory function that creates a fresh environment from a given seed
        n_episodes : int
            Number of evaluation episodes to run
        seed : int
            Base random seed used for reproducible evaluation

        Returns
        -------
        tuple[EvaluationSummary, list[EpisodeEvaluation]]
            Aggregated evaluation metrics and per-episode evaluation results
        """
        raise NotImplementedError("SB3 model not implemented yet.")

    def save(self, path: str | Path) -> None:
        """
        Save the model to disk

        Parameters
        ----------
        path : str | Path
            Destination path where the model should be saved
        """
        raise NotImplementedError("SB3 model not implemented yet.")

    @classmethod
    def load(cls, path: str | Path) -> "SB3Model":
        """
        Load a model instance from disk

        Parameters
        ----------
        path : str | Path
            Path to the saved model file

        Returns
        -------
        SB3Model
            Loaded SB3 model instance
        """
        raise NotImplementedError("SB3 model not implemented yet.")
