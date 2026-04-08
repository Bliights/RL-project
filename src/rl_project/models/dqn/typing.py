from dataclasses import asdict, dataclass


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
