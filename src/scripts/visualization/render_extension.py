import gymnasium as gym

from rl_project.models.dqn.model import DQNModel
from scripts.utils.benchmark_config_extension import (
    EXTENSION_ENV_ID,
    EXTENSION_GREEDY_PASSING_CONFIG,
)

model = DQNModel.load(
    "C:/Users/fanny/OneDrive/Bureau/Cours_CS/RL/RL-project/data/extension/greedy/seed_0/model_dqn_seed_0.pt",
)

env = gym.make(EXTENSION_ENV_ID, render_mode="human", config=EXTENSION_GREEDY_PASSING_CONFIG)

obs, _ = env.reset()
done = False

while not done:
    obs_flat = obs.flatten()
    action = model.act(obs_flat)
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"action={action} | speed={info.get('speed', 0):.1f} | lane={info.get('lane_index', '?')}"
    )
    done = terminated or truncated

env.close()
