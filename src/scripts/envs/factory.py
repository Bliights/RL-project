from scripts.envs.baseline import BASELINE_CONFIG
from scripts.envs.dense import DENSE_CONFIG
from scripts.envs.greedy import GREEDY_CONFIG
from scripts.envs.security import SECURITY_CONFIG
from scripts.envs.typing import EnvType

SHARED_CORE_ENV_ID = "highway-v0"


def get_env_config(env_type: EnvType) -> tuple[str, dict]:
    """
    Return the env id and the config base on the type of environment chosen

    Parameters
    ----------
    env_type : EnvType
        The type of environment chosen

    Returns
    -------
    tuple[str, dict]
        env_id, env_config

    Raises
    ------
    ValueError
        If the environment type doesn't exist
    """
    if env_type == EnvType.BASELINE:
        return SHARED_CORE_ENV_ID, BASELINE_CONFIG

    if env_type == EnvType.GREEDY:
        return SHARED_CORE_ENV_ID, GREEDY_CONFIG

    if env_type == EnvType.SECURITY:
        return SHARED_CORE_ENV_ID, SECURITY_CONFIG

    if env_type == EnvType.DENSE:
        return SHARED_CORE_ENV_ID, DENSE_CONFIG

    raise ValueError(f"Unknown environment type: {env_type}")
