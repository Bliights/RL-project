from scripts.utils.benchmark_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

EXTENSION_ENV_ID = SHARED_CORE_ENV_ID

EXTENSION_GREEDY_PASSING_CONFIG = {
    **SHARED_CORE_CONFIG,
    "lane_change_reward": 0.15,
    "right_lane_reward": -0.1,
}


EXTENSION_SECURITY_PASSING_CONFIG = {
    **EXTENSION_GREEDY_PASSING_CONFIG,
    "lane_change_reward": 0.15,
    "right_lane_reward": -0.1,
    "collision_reward": -3.0,
}
