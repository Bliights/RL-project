from pathlib import Path

from rl_project.models.dqn.typing import DQNConfig

DEFAULT_DQN_CONFIG = DQNConfig(
    gamma=0.99,
    batch_size=64,
    buffer_capacity=50_000,
    update_target_every=500,
    epsilon_start=1.0,
    decrease_epsilon_factor=200,
    epsilon_min=0.05,
    learning_rate=1e-3,
    hidden_size=128,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "training"
DEFAULT_CHECKPOINT_EVERY_EPISODES = 500
DEFAULT_EVAL_EVERY_EPISODES = 500
DEFAULT_EVAL_EPISODES = 10
