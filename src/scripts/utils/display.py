import time

import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output

from rl_project.models.core.base import BaseRLModel


def display_episode(
    env: gym.Env,
    model: BaseRLModel,
    greedy: bool = True,
    sleep_time: float = 0.0,
) -> dict:
    """
    Run and display a full episode using the given model

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment with rendering enabled
    model : BaseRLModel
        Reinforcement learning model used to select actions
    greedy : bool, optional
        Whether to use greedy action selection during the episode
    sleep_time : float, optional
        Delay in seconds between two rendered steps

    Returns
    -------
    dict
        Dictionary containing summary metrics for the displayed episode (episode_reward,
        episode_length, crashed, offroad and mean_speed)
    """
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0
    final_info: dict = {}

    while not done:
        action = model.act(state, greedy=greedy)
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_idx += 1
        state = next_state
        done = terminated or truncated
        final_info = info

        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.imshow(env.render())
        plt.axis("off")
        plt.title(f"Step={step_idx} | Cumulative reward={total_reward:.2f}")
        plt.show()

        if sleep_time > 0.0:
            time.sleep(sleep_time)

    return {
        "episode_reward": float(final_info.get("episode_reward", total_reward)),
        "episode_length": int(final_info.get("episode_length", step_idx)),
        "crashed": bool(final_info.get("crashed", False)),
        "offroad": bool(final_info.get("offroad", False)),
        "mean_speed": float(final_info.get("mean_speed", 0.0)),
    }
