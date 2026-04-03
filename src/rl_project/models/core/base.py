from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    TrainingEpisodeMetrics,
)


class BaseRLModel(ABC):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        """
        Initialize the base RL model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation vector used as input to the model
        n_actions : int
            Number of possible actions the agent can choose from
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action for a given environment state

        Parameters
        ----------
        state : np.ndarray
            Current observation of the environment
        greedy : bool, optional
            To select the action greedily

        Returns
        -------
        int
            Selected action index
        """

    @abstractmethod
    def fit(
        self,
        env: gym.Env,
        output_dir: Path,
        seed: int,
    ) -> list[TrainingEpisodeMetrics]:
        """
        Train the model on the given environment

        Parameters
        ----------
        env : gym.Env
            Gymnasium environment used for training
        output_dir : Path
            Directory where training artifacts such as checkpoints and models may be saved
        seed : int
            Random seed used to ensure reproducibility

        Returns
        -------
        list[TrainingEpisodeMetrics]
            List of episode-level training metrics collected during training
        """

    @abstractmethod
    def evaluate(
        self,
        env_factory: Callable[[int], gym.Env],
        n_episodes: int,
        seed: int,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the model over multiple episodes

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
            An aggregated evaluation summary and a list of per-episode evaluation results
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the model to disk

        Parameters
        ----------
        path : str | Path
            Destination path where the model should be saved
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRLModel":
        """
        Load a model instance from disk

        Parameters
        ----------
        path : str | Path
            Path to the saved model

        Returns
        -------
        BaseRLModel
            Loaded model instance
        """
