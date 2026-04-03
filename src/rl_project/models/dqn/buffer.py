import random

import torch

Transition = tuple[
    torch.Tensor,  # state
    torch.Tensor,  # action
    torch.Tensor,  # reward
    torch.Tensor,  # terminated
    torch.Tensor,  # next_state
]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay buffer

        Parameters
        ----------
        capacity : int
            Maximum number of transitions stored in memory
        """
        self.capacity = capacity
        self.memory: list[Transition | None] = []
        self.position = 0

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        next_state: torch.Tensor,
    ) -> None:
        """
        Store a transition in the replay buffer

        Parameters
        ----------
        state : torch.Tensor
            Tensor representation of the current state
        action : torch.Tensor
            Action taken in the current state
        reward : torch.Tensor
            Reward received after taking the action
        terminated : torch.Tensor
            Termination flag indicating whether the transition leads to a terminal state
        next_state : torch.Tensor
            Tensor representation of the next state
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (
            state,
            action,
            reward,
            terminated,
            next_state,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
    ) -> list[Transition]:
        """
        Sample a batch of transitions uniformly at random

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        list[Transition]
            Randomly sampled batch of transitions
        """
        return random.choices(self.memory, k=batch_size)

    def __len__(self) -> int:
        """
        Return the current number of stored transitions

        Returns
        -------
        int
            Number of transitions currently available in the buffer
        """
        return len(self.memory)
