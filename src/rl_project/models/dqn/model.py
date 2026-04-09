import logging
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
    TrainingState,
    TrainingStepMetrics,
)
from rl_project.models.dqn.buffer import ReplayBuffer
from rl_project.models.dqn.network import QNetwork
from rl_project.models.dqn.typing import DQNConfig

logger = logging.getLogger(__name__)


class DQNModel(BaseRLModel):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: torch.device | str | None = None,
    ) -> None:
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
        device : torch.device | str | None
            CPU or GPU
        """
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, device=device)
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
        self.training_state = TrainingState(
            base_seed=-1,
            completed_steps=-1,
            completed_episodes=0,
            episode=0,
            step_in_episode=-1,
            best_mean_reward=-float("inf"),
            history=[],
        )

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
        ) * np.exp(
            -1.0 * self.training_state.completed_episodes / self.config.decrease_epsilon_factor,
        )

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
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
        done : bool
            Whether the transition leads to a terminal state or limit
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
            torch.tensor([done], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.config.batch_size:
            return float("inf")

        self.training_state.completed_steps += 1

        transitions = self.buffer.sample(self.config.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            done_batch,
            next_state_batch,
        ) = tuple(torch.cat(items).to(self.device) for items in zip(*transitions))

        values = self.q_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = (1.0 - done_batch) * self.target_net(next_state_batch).max(1)[0]
            targets = reward_batch + self.config.gamma * next_state_values

        loss = self.loss_function(values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_state.completed_steps % self.config.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if done:
            self.training_state.completed_episodes += 1

        self.decrease_epsilon()

        return float(loss.detach().cpu().item())

    def _warmup_buffer(
        self,
        env: gym.Env,
        seed: int,
    ) -> int:
        """
        Warmup the buffer before the training

        Parameters
        ----------
        env : gym.Env
            The environment used in the training
        seed : int
            The base seed to use

        Returns
        -------
        int
            The new seed after the buffer start
        """
        new_seed = seed
        state, _ = env.reset(seed=new_seed)
        episode = 0

        while len(self.buffer) < self.config.batch_size:
            action = self.act(state, greedy=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            self.buffer.push(
                torch.tensor(state).unsqueeze(0),
                torch.tensor([[action]], dtype=torch.int64),
                torch.tensor([reward]),
                torch.tensor([terminated or truncated], dtype=torch.int64),
                torch.tensor(next_state).unsqueeze(0),
            )
            state = next_state
            if terminated or truncated:
                episode += 1
                state, _ = env.reset(seed=new_seed + episode)

        logger.info("Warmup of the model buffer finished !")
        return new_seed + episode + 1

    def fit(
        self,
        env: gym.Env,
        output_dir: Path,
        seed: int,
        n_steps: int,
        checkpoint_every_episodes: int,
        training_info: str,
        eval_every_episodes: int,
        eval_episodes: int,
    ) -> list[TrainingStepMetrics]:
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
        n_steps : int
            Number of training environment steps / timesteps
        checkpoint_every_episodes : int
            Save a regular checkpoint every X completed episodes
        training_info : str
            String of the training info
        eval_every_episodes : int
            Run evaluation every X completed episodes
        eval_episodes : int
            Number of greedy evaluation episodes

        Returns
        -------
        list[TrainingStepMetrics]
            Step-level training metrics collected throughout training
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.training_state.base_seed == -1:
            self.training_state.base_seed = self._warmup_buffer(env, seed)

        state, _ = env.reset(
            seed=self.training_state.base_seed + self.training_state.completed_episodes,
        )

        with tqdm(
            range(n_steps),
            desc="Training steps",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="green",
        ) as step_bar:
            for global_step in step_bar:
                self.training_state.step_in_episode += 1

                action = self.act(state, greedy=False)
                next_state, reward, terminated, truncated, info = env.step(action)

                loss = self.update(
                    state=state,
                    action=action,
                    reward=float(reward),
                    done=terminated or truncated,
                    next_state=next_state,
                )

                metrics = TrainingStepMetrics(
                    step=global_step,
                    episode=self.training_state.episode,
                    step_in_episode=self.training_state.step_in_episode,
                    reward=float(reward),
                    crashed=bool(info.get("crashed", False)),
                    offroad=not bool(info.get("on_road", True)),
                    speed=float(info.get("speed", 0.0)),
                )
                self.training_state.history.append(metrics)

                loss_value = loss if np.isfinite(loss) else float("nan")
                step_bar.set_postfix_str(
                    f"ep={self.training_state.episode}  "
                    f"step_ep={self.training_state.step_in_episode}  "
                    f"reward={metrics.reward:.4f}  "
                    f"loss={loss_value:.4f}  "
                    f"epsilon={self.epsilon:.4f}",
                )

                state = next_state

                if terminated or truncated:
                    self.training_state.episode += 1
                    self.training_state.step_in_episode = -1
                    if (
                        checkpoint_every_episodes > 0
                        and self.training_state.completed_episodes % checkpoint_every_episodes == 0
                    ):
                        checkpoint_path = (
                            checkpoint_dir
                            / f"checkpoint_{training_info}_episode_{self.training_state.completed_episodes}.pt"
                        )
                        self.save(checkpoint_path)

                    if (
                        eval_every_episodes > 0
                        and self.training_state.completed_episodes % eval_every_episodes == 0
                    ):
                        eval_seed = (
                            self.training_state.base_seed
                            + 100_000
                            + self.training_state.completed_episodes
                        )
                        summary, _ = self.evaluate(
                            env=env,
                            n_episodes=eval_episodes,
                            seed=eval_seed,
                            verbose=False,
                        )

                        if summary.mean_reward > self.training_state.best_mean_reward:
                            self.training_state.best_mean_reward = summary.mean_reward
                            best_model_path = output_dir / f"model_{training_info}_best.pt"
                            self.save(best_model_path)

                    state, _ = env.reset(
                        seed=self.training_state.base_seed + self.training_state.episode,
                    )

        return self.training_state.history

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int,
        seed: int,
        verbose: bool = False,
    ) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
        """
        Evaluate the trained policy over multiple episodes (It is performed greedily,
        without exploration)

        Parameters
        ----------
        env: gym.Env
            Evaluation environment
        n_episodes : int
            Number of evaluation episodes to run
        seed : int
            Base random seed used to derive episode seeds
        verbose : bool
            To enable or disable the display

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

        iterator = (
            tqdm(
                range(n_episodes),
                desc="Evaluation Episodes",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
                colour="blue",
            )
            if verbose
            else range(n_episodes)
        )

        for episode_idx in iterator:
            current_seed = seed + episode_idx
            state, _ = env.reset(seed=current_seed)
            done = False
            final_info = {}

            while not done:
                action = self.act(state, greedy=True)
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_info = info

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

            if verbose:
                iterator.set_postfix_str(
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
        if verbose:
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
            "device": str(self.device),
            "buffer": self.buffer.state_dict(),
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_state": self.training_state.to_dict(),
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
            device=checkpoint["device"],
        )
        model.buffer.load_state_dict(checkpoint["buffer"])
        model.q_net.load_state_dict(checkpoint["q_net"])
        model.target_net.load_state_dict(checkpoint["target_net"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        model.epsilon = float(checkpoint["epsilon"])
        model.training_state = TrainingState.from_dict(checkpoint["training_state"])

        model.q_net.to(model.device)
        model.target_net.to(model.device)
        return model


class DoubleDQNModel(DQNModel):
    """
    Double DQN model — identical to DQN except for the target computation.

    In standard DQN, the same network selects AND evaluates the next action,
    leading to overestimation bias. Double DQN fixes this by:
    - Using q_net to SELECT the best next action
    - Using target_net to EVALUATE that action
    """

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> float:
        """
        Store one transition and perform a double DQN optimization step

        Parameters
        ----------
        state : np.ndarray
            Current state
        action : int
            Action taken in the current state
        reward : float
            Reward received after executing the action
        done : bool
            Whether the transition leads to a terminal state or limit
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
            torch.tensor([done], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.config.batch_size:
            return float("inf")

        self.training_state.completed_steps += 1

        transitions = self.buffer.sample(self.config.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            done_batch,
            next_state_batch,
        ) = tuple(torch.cat(items).to(self.device) for items in zip(*transitions))

        values = self.q_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # Here for the double DQN, the q_net chose the best next action
            # and target_net evaluate that action to compute the target
            next_actions = self.q_net(next_state_batch).argmax(1, keepdim=True)
            next_state_values = (1.0 - done_batch) * self.target_net(next_state_batch).gather(
                1,
                next_actions,
            ).squeeze(1)
            targets = reward_batch + self.config.gamma * next_state_values

        loss = self.loss_function(values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_state.completed_steps % self.config.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if done:
            self.training_state.completed_episodes += 1

        self.decrease_epsilon()

        return float(loss.detach().cpu().item())
