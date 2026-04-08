from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from rl_project.benchmark.benchmark import HighwayBenchmark
from rl_project.benchmark.typing import BenchmarkConfig
from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    ModelType,
    TrainingState,
    TrainingStepMetrics,
)
from rl_project.models.sb3.callback import SB3TrainingCallback
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
        self.type = ModelType.SB3
        self.model: DQN | None = None
        self.training_state = TrainingState(
            base_seed=-1,
            completed_steps=-1,
            completed_episodes=0,
            episode=0,
            step_in_episode=-1,
            best_mean_reward=-float("inf"),
            history=[],
        )

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
        action, _ = self.model.predict(
            observation=state,
            deterministic=greedy,
        )
        return int(action)

    def _build_model(self, env: gym.Env, seed: int) -> None:
        """
        Build the sb3 model for the training

        Parameters
        ----------
        env : gym.Env
            The training env
        seed : int
            the seed
        """
        self.model = DQN(
            policy=self.config.policy,
            env=DummyVecEnv([lambda: env]),
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            policy_kwargs={"net_arch": self.config.net_arch},
            verbose=self.config.verbose,
            seed=seed,
            device=str(self.device),
        )

    def _make_eval_env(self, env: gym.Env) -> tuple[gym.Env, bool]:
        """
        Create the evaluation environment

        Returns
        -------
        tuple[gym.Env, bool]
            evaluation environment, whether the env should be closed by the caller
        """
        if self.config.eval_env_id is not None and self.config.eval_env_config is not None:
            benchmark = HighwayBenchmark(
                config=BenchmarkConfig(
                    env_id=self.config.eval_env_id,
                    env_config=self.config.eval_env_config,
                ),
                render_mode=self.config.eval_render_mode,
            )
            return benchmark.make_env(), True

        return env, False

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
        training_info : str
            String of the training info
        eval_every_episodes : int
            Run evaluation every X completed episodes
        eval_episodes : int
            Number of greedy evaluation episodes

        Returns
        -------
        list[TrainingStepMetrics]
            Step-level training metrics collected throughout training
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            self._build_model(env=env, seed=seed)
        else:
            self.model.set_env(DummyVecEnv([lambda: env]))

        eval_env, should_close = self._make_eval_env(env)

        callback = SB3TrainingCallback(
            owner=self,
            eval_env=eval_env,
            output_dir=output_dir,
            seed=seed,
            checkpoint_every_episodes=checkpoint_every_episodes,
            training_info=training_info,
            eval_every_episodes=eval_every_episodes,
            eval_episodes=eval_episodes,
        )
        try:
            self.model.learn(
                total_timesteps=n_steps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=False,
            )
        finally:
            if should_close:
                eval_env.close()

        return self.training_state.history

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
        rewards = []
        lengths = []
        crashes = []
        offroads = []
        speeds = []
        episodes = []

        iterator = (
            tqdm(
                range(n_episodes),
                desc="Evaluation Episodes",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
                colour="blue",
            )
            if verbose
            else range(n_episodes)
        )

        for episode_idx in iterator:
            current_seed = seed + episode_idx
            state, _ = env.reset(seed=current_seed)
            done = False
            final_info = {}

            while not done:
                action = self.act(state, greedy=True)
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_info = info

            result = EpisodeEvaluation(
                episode=episode_idx,
                seed=current_seed,
                reward=float(final_info.get("episode_reward", 0.0)),
                length=int(final_info.get("episode_length", 0)),
                crashed=bool(final_info.get("crashed", False)),
                offroad=bool(final_info.get("offroad", False)),
                mean_speed=float(final_info.get("mean_speed", 0.0)),
            )
            episodes.append(result)

            rewards.append(result.reward)
            lengths.append(result.length)
            crashes.append(float(result.crashed))
            offroads.append(float(result.offroad))
            speeds.append(result.mean_speed)

            if verbose:
                iterator.set_postfix_str(
                    f"reward={result.reward:.4f}  "
                    f"length={result.length:d}  "
                    f"crashed={result.crashed}  "
                    f"offroad={result.offroad}",
                )

        summary = EvaluationSummary(
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_episode_length=float(np.mean(lengths)),
            crash_rate=float(np.mean(crashes)),
            offroad_rate=float(np.mean(offroads)),
            mean_speed=float(np.mean(speeds)),
            n_episodes=n_episodes,
        )
        return summary, episodes

    def save(self, path: str | Path) -> None:
        """
        Save the model to disk

        Parameters
        ----------
        path : str | Path
            Destination path where the model should be saved
        """
        if self.model is None:
            raise RuntimeError("SB3 model has not been initialized or loaded.")

        path = Path(path)
        zip_path = path.with_suffix(".zip")
        replay_buffer_path = path.with_suffix(".replay.pkl")

        self.model.save(zip_path)
        self.model.save_replay_buffer(replay_buffer_path)

        payload = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "config": self.config.to_dict(),
            "device": str(self.device),
            "zip_name": zip_path.name,
            "replay_buffer_name": replay_buffer_path.name,
            "training_state": self.training_state.to_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "SB3Model":
        """
        Load a model instance from disk

        Parameters
        ----------
        path : str | Path
            Path to the saved model file (.pt file)

        Returns
        -------
        SB3Model
            Loaded SB3 model instance
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        model = cls(
            obs_dim=int(payload["obs_dim"]),
            n_actions=int(payload["n_actions"]),
            config=SB3Config(**payload["config"]),
            device=payload["device"],
        )

        zip_path = path.with_suffix(".zip")
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing SB3 weights file: {zip_path}")

        model.model = DQN.load(
            path=zip_path,
            device=str(model.device),
        )

        replay_buffer_path = path.with_suffix(".replay.pkl")
        if not replay_buffer_path.exists():
            raise FileNotFoundError(f"Missing SB3 buffer file: {replay_buffer_path}")
        model.model.load_replay_buffer(replay_buffer_path)

        model.training_state = TrainingState.from_dict(payload["training_state"])

        return model
