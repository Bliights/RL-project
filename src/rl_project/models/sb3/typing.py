from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SB3Config:
    policy: str
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    target_update_interval: int
    exploration_fraction: float
    exploration_initial_eps: float
    exploration_final_eps: float
    net_arch: list[int]
    verbose: int
    eval_env_id: str
    eval_env_config: dict
    eval_render_mode: str

    def to_dict(self) -> dict:
        """
        Return a dictionary of the class instance

        Returns
        -------
        dict
            The dictionary of the class instance
        """
        return asdict(self)
