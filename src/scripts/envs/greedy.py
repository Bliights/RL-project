from scripts.envs.baseline import BASELINE_CONFIG

GREEDY_CONFIG = {
    **BASELINE_CONFIG,
    "lane_change_reward": 0.15,
    "right_lane_reward": -0.1,
}
