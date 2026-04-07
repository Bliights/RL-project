import numpy as np
import torch

from rl_project.models.dqn.model import DQNModel


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
        terminated: bool,
        next_state: np.ndarray,
    ) -> float:
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
            # Double DQN : q_net choisit l'action, target_net l'évalue
            next_actions = self.q_net(next_state_batch).argmax(1, keepdim=True)
            next_state_values = (1.0 - terminated_batch) * self.target_net(
                next_state_batch,
            ).gather(1, next_actions).squeeze(1)
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
