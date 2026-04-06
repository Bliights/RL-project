from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    TrainingStepMetrics,
)
from rl_project.models.sb3.typing import SB3Config


class SB3Model(BaseRLModel):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: SB3Config,
        device: torch.device | str | None = None,
    ) -> None:
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
        device : torch.device | str | None
            CPU or GPU
        """
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, device=device)
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

    def fit(
        self,
        env: gym.Env,
        output_dir: Path,
        seed: int,
        n_steps: int,
        checkpoint_every_episodes: int,
        eval_every_episodes: int,
        eval_episodes: int,
    ) -> list[TrainingStepMetrics]:
        """
        Train the SB3 agent on the given environment

        Parameters
        ----------
        env : gym.Env
            Training environment
        output_dir : Path
            Directory where checkpoints are saved
        seed : int
            Base random seed used to initialize episodes reproducibly
        n_steps : int
            Number of training environment steps / timesteps
        checkpoint_every_episodes : int
            Save a regular checkpoint every X completed episodes
        eval_every_episodes : int
            Run evaluation every X completed episodes
        eval_episodes : int
            Number of greedy evaluation episodes

        Returns
        -------
        list[TrainingStepMetrics]
            Step-level training metrics collected throughout training
        """
        raise NotImplementedError("SB3 model not implemented yet.")

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int,
        seed: int,
        verbose: bool = False,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the trained policy over multiple episodes (It is performed greedily,
        without exploration)

        Parameters
        ----------
        env: gym.Env
            Evaluation environment
        n_episodes : int
            Number of evaluation episodes to run
        seed : int
            Base random seed used to derive episode seeds
        verbose : bool
            To enable or disable the display

        Returns
        -------
        tuple[EvaluationSummary, list[EpisodeEvaluation]]
            A tuple containing the aggregated evaluation summary and the list
            of per-episode evaluation results
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
