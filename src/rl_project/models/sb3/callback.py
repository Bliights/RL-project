from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

from rl_project.models.core.typing import TrainingStepMetrics

if TYPE_CHECKING:
    from pathlib import Path

    import gymnasium as gym

    from rl_project.models.sb3.model import SB3Model


class SB3TrainingCallback(BaseCallback):
    def __init__(
        self,
        owner: SB3Model,
        eval_env: gym.Env,
        output_dir: Path,
        seed: int,
        checkpoint_every_episodes: int,
        training_info: str,
        eval_every_episodes: int,
        eval_episodes: int,
    ) -> None:
        """
        Custom callback for monitoring and managing SB3 training.

        Parameters
        ----------
        owner : SB3Model
            The SB3 model being trained
        eval_env : gym.Env
            Environment used for evaluating the model
        output_dir : Path
            Directory where checkpoints and the best model are saved
        seed : int
            Random seed used for reproducibility
        checkpoint_every_episodes : int
            Save a regular checkpoint every X completed episodes
        training_info : str
            String of the training info
        eval_every_episodes : int
            Run evaluation every X completed episodes
        eval_episodes : int
            Number of greedy evaluation episodes
        """
        super().__init__()
        self.owner = owner
        self.eval_env = eval_env
        self.output_dir = output_dir
        self.seed = seed
        self.checkpoint_every_episodes = checkpoint_every_episodes
        self.training_info = training_info
        self.eval_every_episodes = eval_every_episodes
        self.eval_episodes = eval_episodes

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Method called at every training step

        Returns
        -------
        bool
            Always returns `True` to continue training
        """
        info = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        self.owner.training_state.completed_steps += 1
        self.owner.training_state.step_in_episode += 1

        metrics = TrainingStepMetrics(
            step=self.owner.training_state.completed_steps,
            episode=self.owner.training_state.episode,
            step_in_episode=self.owner.training_state.step_in_episode,
            reward=reward,
            crashed=bool(info.get("crashed", False)),
            offroad=bool(info.get("offroad", False)),
            speed=float(info.get("speed", 0.0)),
        )
        self.owner.training_state.history.append(metrics)

        if done:
            self.owner.training_state.completed_episodes += 1
            self.owner.training_state.episode += 1
            self.owner.training_state.step_in_episode = -1

            if (
                self.checkpoint_every_episodes > 0
                and self.owner.training_state.completed_episodes % self.checkpoint_every_episodes
                == 0
            ):
                checkpoint_path = (
                    self.checkpoint_dir
                    / f"checkpoint_{self.training_info}_episode_{self.owner.training_state.completed_episodes}.pt"
                )
                self.owner.save(checkpoint_path)

            if (
                self.eval_every_episodes > 0
                and self.owner.training_state.completed_episodes % self.eval_every_episodes == 0
            ):
                eval_seed = self.seed + 100_000 + self.owner.training_state.completed_episodes
                summary, _ = self.owner.evaluate(
                    env=self.eval_env,
                    n_episodes=self.eval_episodes,
                    seed=eval_seed,
                    verbose=False,
                )
                if summary.mean_reward > self.owner.training_state.best_mean_reward:
                    self.owner.training_state.best_mean_reward = summary.mean_reward
                    best_path = self.output_dir / f"model_{self.training_info}_best.pt"
                    self.owner.save(best_path)

        return True
