from pathlib import Path

from rl_project.models.dqn.typing import DQNConfig
from rl_project.models.sb3.typing import SB3Config
from scripts.utils.benchmark_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

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

DEFAULT_SB3_CONFIG = SB3Config(
    policy="MlpPolicy",
    learning_rate=5e-4,
    buffer_size=50_000,
    learning_starts=2_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.15,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    net_arch=[256, 256],
    verbose=0,
    eval_env_id=SHARED_CORE_ENV_ID,
    eval_env_config=SHARED_CORE_CONFIG,
    eval_render_mode="rgb_array",
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "training"
DEFAULT_CHECKPOINT_EVERY_EPISODES = 500
DEFAULT_EVAL_EVERY_EPISODES = 500
DEFAULT_EVAL_EPISODES = 10
