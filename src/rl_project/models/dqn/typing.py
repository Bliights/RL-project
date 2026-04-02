from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DQNConfig:
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 50_000
    update_target_every: int = 500
    epsilon_start: float = 1.0
    decrease_epsilon_factor: float = 200
    epsilon_min: float = 0.05
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_episodes: int = 300
    checkpoint_every: int = 25
    eval_episodes: int = 50

    def to_dict(self) -> dict:
        """
        Return a dictionary of the class instance

        Returns
        -------
        dict
            The dictionary of the class instance
        """
        return asdict(self)
