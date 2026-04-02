from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box


class FlattenObservationWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Wrapper that flattens high-dimensional observations into a 1D vector

        Parameters
        ----------
        env : gym.Env
            Environment with a Box observation space

        Raises
        ------
        TypeError
            If the observation space is not of type gymnasium.spaces.Box
        """
        super().__init__(env)

        if not isinstance(env.observation_space, Box):
            raise TypeError("Observation space must be a Box.")

        flat_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and return a flattened observation

        Parameters
        ----------
        seed : int | None, optional
            Random seed for environment reset
        options : dict[str, Any] | None, optional
            Additional reset options

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Flattened observation and info dictionary
        """
        observation, info = self.env.reset(seed=seed, options=options)
        return self._flatten(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Perform one environment step and return a flattened observation

        Parameters
        ----------
        action : int
            Action selected by the agent

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            Flattened observation, reward, termination flag, truncation flag, and info dictionary.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(observation), float(reward), terminated, truncated, info

    @staticmethod
    def _flatten(observation: np.ndarray) -> np.ndarray:
        """
        Flatten an observation into a 1D vector

        Parameters
        ----------
        observation : np.ndarray
            Original observation

        Returns
        -------
        np.ndarray
            Flattened observation
        """
        return np.asarray(observation, dtype=np.float32).reshape(-1)


class EpisodeMetricsWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Wrapper that tracks and enriches episode-level metrics

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        """
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.step_speeds = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and episode statistics

        Parameters
        ----------
        seed : int | None, optional
            Random seed for environment reset
        options : dict[str, Any] | None, optional
            Additional reset options

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Initial observation and info dictionary
        """
        self.episode_reward = 0.0
        self.episode_length = 0
        self.step_speeds = []
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Perform one step and update episode metrics

        Parameters
        ----------
        action : int
            Action selected by the agent

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            Observation, reward, termination flag, truncation flag, and enriched info dictionary
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.episode_reward += float(reward)
        self.episode_length += 1
        self.step_speeds.append(float(info.get("speed", 0.0)))

        if terminated or truncated:
            enriched_info = dict(info)
            enriched_info["episode_reward"] = self.episode_reward
            enriched_info["episode_length"] = self.episode_length
            enriched_info["crashed"] = bool(info.get("crashed", False))
            enriched_info["offroad"] = not bool(info.get("on_road", True))
            enriched_info["mean_speed"] = (
                float(np.mean(self.step_speeds)) if self.step_speeds else 0.0
            )
            return observation, float(reward), terminated, truncated, enriched_info

        return observation, float(reward), terminated, truncated, info
