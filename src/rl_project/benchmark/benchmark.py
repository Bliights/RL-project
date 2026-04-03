import gymnasium as gym
import highway_env  # noqa: F401

from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.benchmark.wrappers import EpisodeMetricsWrapper, FlattenObservationWrapper


class HighwayBenchmark:
    def __init__(
        self,
        config: BenchmarkConfig,
        render_mode: str = "rgb_array",
    ) -> None:
        """
        Benchmark wrapper for creating and configuring Highway environments

        Parameters
        ----------
        config : BenchmarkConfig
            Configuration object containing the Gymnasium environment ID and a dictionary
            of environment parameters
        render_mode : str, optional
            Rendering mode passed to the environment
        """
        self.config = config
        self.render_mode = render_mode

    def make_env(self, seed: int | None = None) -> gym.Env:
        """
        Create a wrapped Gymnasium environment instance

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility

        Returns
        -------
        gym.Env
            A fully initialized and wrapped Gymnasium environment
        """
        env = gym.make(
            self.config.env_id,
            config=self.config.env_config,
            render_mode=self.render_mode,
        )
        env = FlattenObservationWrapper(env)
        env = EpisodeMetricsWrapper(env)

        if seed is not None:
            env.reset(seed=seed)

        return env
