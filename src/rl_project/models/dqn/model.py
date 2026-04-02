import logging
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from rl_project.models.core.base import BaseRLModel
from rl_project.models.core.typing import (
    EpisodeEvaluation,
    EvaluationSummary,
    ModelType,
    TrainingEpisodeMetrics,
)
from rl_project.models.dqn.buffer import ReplayBuffer
from rl_project.models.dqn.network import QNetwork
from rl_project.models.dqn.typing import DQNConfig

logger = logging.getLogger(__name__)


class DQNModel(BaseRLModel):
    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig) -> None:
        """
        Initialize the DQN model and its training components

        Parameters
        ----------
        obs_dim : int
            Dimension of the input observation vector
        n_actions : int
            Number of possible discrete actions
        config : DQNConfig
            DQN hyperparameter configuration
        """
        super().__init__(obs_dim=obs_dim, n_actions=n_actions)
        self.config = config
        self.type = ModelType.DQN

        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.q_net = QNetwork(obs_dim, self.config.hidden_size, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, self.config.hidden_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(),
            lr=self.config.learning_rate,
        )

        self.epsilon = self.config.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
        self.last_loss = None

    def get_q(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for a given state

        Parameters
        ----------
        state : np.ndarray
            Environment observation

        Returns
        -------
        np.ndarray
            Estimated Q-values for all available actions
        """
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.squeeze(0).detach().cpu().numpy()

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action using an epsilon-greedy policy. When :
            - greedy=False, the action is selected using the current epsilon value for exploration
            - greedy=True, the action with the highest estimated Q-value is always selected

        Parameters
        ----------
        state : np.ndarray
            Current environment observation
        greedy : bool, optional
            Whether to disable exploration and act greedily

        Returns
        -------
        int
            Selected action index
        """
        epsilon = 0.0 if greedy else self.epsilon
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.get_q(state)))

    def decrease_epsilon(self) -> None:
        """
        Update the exploration rate according to an exponential decay schedule
        """
        self.epsilon = self.config.epsilon_min + (
            self.config.epsilon_start - self.config.epsilon_min
        ) * np.exp(-1.0 * self.n_eps / self.config.decrease_epsilon_factor)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_state: np.ndarray,
    ) -> float:
        """
        Store one transition and perform a DQN optimization step

        Parameters
        ----------
        state : np.ndarray
            Current state
        action : int
            Action taken in the current state
        reward : float
            Reward received after executing the action
        terminated : bool
            Whether the transition leads to a terminal state
        next_state : np.ndarray
            Next observed state

        Returns
        -------
        float
            Training loss value for the update step
        """
        self.buffer.push(
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.config.batch_size:
            return float("inf")

        transitions = self.buffer.sample(self.config.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            terminated_batch,
            next_state_batch,
        ) = tuple(torch.cat(items).to(self.device) for items in zip(*transitions))

        values = self.q_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = (1.0 - terminated_batch) * self.target_net(next_state_batch).max(1)[
                0
            ]
            targets = reward_batch + self.config.gamma * next_state_values

        loss = self.loss_function(values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.config.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss

    def fit(self, env: gym.Env, output_dir: Path, seed: int) -> list[TrainingEpisodeMetrics]:
        """
        Train the DQN agent on the given environment

        Parameters
        ----------
        env : gym.Env
            Training environment
        output_dir : Path
            Directory where checkpoints are saved
        seed : int
            Base random seed used to initialize episodes reproducibly

        Returns
        -------
        list[TrainingEpisodeMetrics]
            Episode-level training metrics collected throughout training
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        history = []

        with tqdm(
            range(1, self.config.num_episodes + 1),
            desc="Training Episodes",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="green",
        ) as episode_bar:
            for episode in episode_bar:
                state, _ = env.reset(seed=seed + episode - 1)
                done = False
                losses = []
                final_info = {}

                while not done:
                    action = self.act(state, greedy=False)
                    next_state, reward, terminated, truncated, info = env.step(action)

                    loss = self.update(
                        state=state,
                        action=action,
                        reward=float(reward),
                        terminated=terminated,
                        next_state=next_state,
                    )
                    if np.isfinite(loss):
                        losses.append(loss)

                    state = next_state
                    done = terminated or truncated
                    final_info = info

                metrics = TrainingEpisodeMetrics(
                    episode=episode,
                    reward=float(final_info.get("episode_reward", 0.0)),
                    length=int(final_info.get("episode_length", 0)),
                    crashed=int(bool(final_info.get("crashed", False))),
                    offroad=int(bool(final_info.get("offroad", False))),
                    mean_speed=float(final_info.get("mean_speed", 0.0)),
                    epsilon=float(self.epsilon),
                    loss=float(np.mean(losses)) if losses else None,
                    total_steps=int(self.n_steps),
                )
                history.append(metrics)

                if episode % self.config.checkpoint_every == 0:
                    checkpoint_path = (
                        output_dir / f"checkpoint_{self.type}_seed_{seed}_ep_{episode}.pt"
                    )
                    self.save(checkpoint_path)

                loss_value = metrics.loss if metrics.loss is not None else float("nan")
                episode_bar.set_postfix_str(
                    f"reward={metrics.reward:.4f}  "
                    f"loss={loss_value:.4f}  "
                    f"epsilon={metrics.epsilon:.4f}",
                )

        return history

    def evaluate(
        self,
        env_factory: Callable[[int], gym.Env],
        n_episodes: int,
        seed: int,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the trained policy over multiple episodes (It is performed greedily,
        without exploration)

        Parameters
        ----------
        env_factory : Callable[[int], gym.Env]
            Factory function that creates a new environment from a given seed
        n_episodes : int
            Number of evaluation episodes to run
        seed : int
            Base random seed used to derive episode seeds

        Returns
        -------
        tuple[EvaluationSummary, list[EpisodeEvaluation]]
            A tuple containing the aggregated evaluation summary and the list
            of per-episode evaluation results
        """
        rewards = []
        lengths = []
        crashes = []
        offroads = []
        speeds = []
        episodes = []

        with tqdm(
            range(n_episodes),
            desc="Evaluation Episodes",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="blue",
        ) as episode_bar:
            for episode_idx in episode_bar:
                current_seed = seed + episode_idx
                env = env_factory(current_seed)

                state, _ = env.reset(seed=current_seed)
                done = False
                final_info: dict = {}

                while not done:
                    action = self.act(state, greedy=True)
                    state, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    final_info = info

                env.close()

                result = EpisodeEvaluation(
                    episode=episode_idx,
                    seed=current_seed,
                    reward=float(final_info.get("episode_reward", 0.0)),
                    length=int(final_info.get("episode_length", 0)),
                    crashed=bool(final_info.get("crashed", False)),
                    offroad=bool(final_info.get("offroad", False)),
                    mean_speed=float(final_info.get("mean_speed", 0.0)),
                )
                episodes.append(result)

                rewards.append(result.reward)
                lengths.append(result.length)
                crashes.append(float(result.crashed))
                offroads.append(float(result.offroad))
                speeds.append(result.mean_speed)

                episode_bar.set_postfix_str(
                    f"reward={result.reward:.4f}  "
                    f"length={result.length:d}  "
                    f"crashed={result.crashed}  "
                    f"offroad={result.offroad}",
                )

        summary = EvaluationSummary(
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_episode_length=float(np.mean(lengths)),
            crash_rate=float(np.mean(crashes)),
            offroad_rate=float(np.mean(offroads)),
            mean_speed=float(np.mean(speeds)),
            n_episodes=n_episodes,
        )
        logger.info(f"Evaluation completed: mean_reward={summary.mean_reward:.4f}")

        return summary, episodes

    def save(self, path: str | Path) -> None:
        """
        Save the DQN model and optimizer state to disk

        Parameters
        ----------
        path : str | Path
            Destination path of the checkpoint file
        """
        checkpoint = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "config": self.config.to_dict(),
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "n_steps": self.n_steps,
            "n_eps": self.n_eps,
            "last_loss": self.last_loss,
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "DQNModel":
        """
        Load a DQN model from a checkpoint file

        Parameters
        ----------
        path : str | Path
            Path to the saved checkpoint

        Returns
        -------
        DQNModel
            Reconstructed DQN model
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)

        model = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            config=DQNConfig(**checkpoint["config"]),
        )
        model.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        model.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.epsilon = float(checkpoint["epsilon"])
        model.n_steps = int(checkpoint["n_steps"])
        model.n_eps = int(checkpoint["n_eps"])
        model.last_loss = checkpoint["last_loss"]

        model.q_net.to(model.device)
        model.target_net.to(model.device)
        return model
