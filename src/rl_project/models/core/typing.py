from dataclasses import asdict, dataclass
from enum import StrEnum


@dataclass(slots=True)
class TrainingStepMetrics:
    step: int
    episode: int
    step_in_episode: int
    reward: float
    crashed: bool
    offroad: bool
    speed: float

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
class EpisodeEvaluation:
    episode: int
    seed: int
    reward: float
    length: int
    crashed: bool
    offroad: bool
    mean_speed: float

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
class EvaluationSummary:
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    crash_rate: float
    offroad_rate: float
    mean_speed: float
    n_episodes: int

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
        """
        Return a dictionary of the class instance

        Returns
        -------
        dict
            The dictionary of the class instance
        """
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
        """
        Instanciate a TrainingState from a dictionnary

        Parameters
        ----------
        data : dict
            The dictionnary to use

        Returns
        -------
        TrainingState
            The instance of TrainingState loaded
        """
        return cls(
            base_seed=data["base_seed"],
            completed_steps=data["completed_steps"],
            completed_episodes=data["completed_episodes"],
            episode=data["episode"],
            step_in_episode=data["step_in_episode"],
            best_mean_reward=data["best_mean_reward"],
            history=[TrainingStepMetrics(**step) for step in data["history"]],
        )


class ModelType(StrEnum):
    DQN = "dqn"
    SB3 = "sb3"
    DOUBLE_DQN = "2dqn"
