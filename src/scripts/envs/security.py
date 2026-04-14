from scripts.envs.greedy import GREEDY_CONFIG

SECURITY_CONFIG = {
    **GREEDY_CONFIG,
    "lane_change_reward": 0.15,
    "right_lane_reward": -0.1,
    "collision_reward": -3.0,
}
