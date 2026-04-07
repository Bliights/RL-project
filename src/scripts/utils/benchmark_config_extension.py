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


EXTENSION_DENSE_TRAFFIC_CONFIG = {
    **EXTENSION_GREEDY_PASSING_CONFIG,  # hérite du greedy
    "vehicles_count": 70,  # plus de voitures
    "vehicles_density": 1.5,  # trafic plus dense
    "high_speed_reward": 1.2,  # baseline=0.7, on encourage plus la vitesse
    "lane_change_reward": 0.2,  # on augmente un peu le bonus dépassement
}
