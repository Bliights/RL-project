from dataclasses import asdict, dataclass

from rl_project.models.core.typing import TrainingStepMetrics


@dataclass(slots=True)
class DQNConfig:
    gamma: float
    batch_size: int
    buffer_capacity: int
    update_target_every: int
    epsilon_start: float
    decrease_epsilon_factor: float
    epsilon_min: float
    learning_rate: float
    hidden_size: int

    def to_dict(self) -> dict:
        """
        Return a dictionary of the class instance

        Returns
        -------
        dict
            The dictionary of the class instance
        """
        return asdict(self)


@dataclass(slots=True)
class TrainingState:
    base_seed: int
    completed_steps: int
    completed_episodes: int
    episode: int
    step_in_episode: int
    best_mean_reward: float
    history: list[TrainingStepMetrics]

    def to_dict(self) -> dict:
        return {
            "base_seed": self.base_seed,
            "completed_steps": self.completed_steps,
            "completed_episodes": self.completed_episodes,
            "episode": self.episode,
            "step_in_episode": self.step_in_episode,
            "best_mean_reward": self.best_mean_reward,
            "history": [step.to_dict() for step in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingState":
        return cls(
            base_seed=data["base_seed"],
            completed_steps=data["completed_steps"],
            completed_episodes=data["completed_episodes"],
            episode=data["episode"],
            step_in_episode=data["step_in_episode"],
            best_mean_reward=data["best_mean_reward"],
            history=[TrainingStepMetrics(**step) for step in data["history"]],
        )
