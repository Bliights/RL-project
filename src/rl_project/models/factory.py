from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import ModelType
from rl_project.models.dqn.model import DQNModel
from rl_project.models.dqn.typing import DQNConfig
from rl_project.models.sb3.model import SB3Model


def build_model(
    model_type: ModelType,
    obs_dim: int,
    n_actions: int,
    config: DQNConfig,
) -> BaseRLModel:
    """
    Build a RL model from the requested model type

    Parameters
    ----------
    model_type : ModelType
        Type of model to instantiate
    obs_dim : int
        Dimension of the input observation vector
    n_actions : int
        Number of discrete actions available in the environment
    config : DQNConfig
        Configuration object used to initialize the model

    Returns
    -------
    BaseRLModel
        Instantiated RL model

    Raises
    ------
    ValueError
        If the provided model type is not supported
    """
    if model_type == ModelType.DQN:
        return DQNModel(
            obs_dim=obs_dim,
            n_actions=n_actions,
            config=config,
        )

    if model_type == ModelType.SB3:
        return SB3Model(
            obs_dim=obs_dim,
            n_actions=n_actions,
            config=config,
        )

    raise ValueError(f"Unknown model type: {model_type}")
