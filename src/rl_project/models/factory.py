from pathlib import Path

from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import ModelType
from rl_project.models.dqn.model import DQNModel
from rl_project.models.dqn.typing import DQNConfig
from rl_project.models.sb3.model import SB3Model
from rl_project.models.sb3.typing import SB3Config


def build_model(
    model_type: ModelType,
    obs_dim: int,
    n_actions: int,
    config: DQNConfig | SB3Config,
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
    config : DQNConfig | SB3Config
        Configuration object used to initialize the model

    Returns
    -------
    BaseRLModel
        Instantiated RL model

    Raises
    ------
    TypeError
        If the config is not adapted to the model
    ValueError
        If the provided model type is not supported
    """
    if model_type == ModelType.DQN:
        if not isinstance(config, DQNConfig):
            raise TypeError("DQN model requires a DQNConfig.")
        return DQNModel(
            obs_dim=obs_dim,
            n_actions=n_actions,
            config=config,
        )

    if model_type == ModelType.SB3:
        if not isinstance(config, SB3Config):
            raise TypeError("SB3 model requires a SB3Config.")
        return SB3Model(
            obs_dim=obs_dim,
            n_actions=n_actions,
            config=config,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def load_model(
    model_type: ModelType,
    model_path: str | Path,
) -> BaseRLModel:
    """
    Load the RL model from the path

    Parameters
    ----------
    model_type : ModelType
        Type of model to instantiate
    model_path : str | Path
        Path of the model file

    Returns
    -------
    BaseRLModel
        Instantiated RL model

    Raises
    ------
    FileNotFoundError
        If the provided model file is not found
    ValueError
        If the provided model type is not supported
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == ModelType.DQN:
        return DQNModel.load(model_path)

    if model_type == ModelType.SB3:
        return SB3Model.load(model_path)

    raise ValueError(f"Unknown model type: {model_type}")
