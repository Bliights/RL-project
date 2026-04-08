from scripts.envs.greedy import GREEDY_CONFIG

DENSE_CONFIG = {
    **GREEDY_CONFIG,
    "vehicles_count": 70,  # More vehicles
    "vehicles_density": 1.5,  # Higher vehicles density
    "high_speed_reward": 1.2,  # More reward for high speed
    "lane_change_reward": 0.2,  # More reward for lane change
}
