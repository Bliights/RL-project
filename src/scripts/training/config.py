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
    num_episodes=12000,
    checkpoint_every=50,
    eval_episodes=50,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "training"
