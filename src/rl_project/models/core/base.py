from abc import ABC, abstractmethod
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    TrainingStepMetrics,
)


class BaseRLModel(ABC):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Initialize the base RL model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation vector used as input to the model
        n_actions : int
            Number of possible actions the agent can choose from
        device : torch.device | str | None
            CPU or GPU
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        """
        Get the good device based on the user machine

        Parameters
        ----------
        device : torch.device | str | None
            The requested device

        Returns
        -------
        torch.device
            The best device found
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")

        return device

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
        n_steps: int,
        checkpoint_every_episodes: int,
        training_info: str,
        eval_every_episodes: int,
        eval_episodes: int,
    ) -> list[TrainingStepMetrics]:
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
        n_steps : int
            Number of training environment steps / timesteps
        checkpoint_every_episodes : int
            Save a regular checkpoint every X completed episodes
        training_info : str
            String of the training info
        eval_every_episodes : int
            Run evaluation every X completed episodes
        eval_episodes : int
            Number of greedy evaluation episodes

        Returns
        -------
        list[TrainingStepMetrics]
            List of episode-level training metrics collected during training
        """

    @abstractmethod
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int,
        seed: int,
        verbose: bool,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the model over multiple episodes

        Parameters
        ----------
        env: gym.Env
            Evaluation environment
        n_episodes : int
            Number of evaluation episodes to run
        seed : int
            Base random seed used for reproducible evaluation
        verbose : bool
            To enable or disable the display

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
