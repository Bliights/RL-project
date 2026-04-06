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


class ModelType(StrEnum):
    DQN = "dqn"
    SB3 = "sb3"
