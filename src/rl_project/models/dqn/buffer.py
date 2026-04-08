import random

import torch

Transition = tuple[
    torch.Tensor,  # state
    torch.Tensor,  # action
    torch.Tensor,  # reward
    torch.Tensor,  # done
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
        done: torch.Tensor,
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
        done : torch.Tensor
            Termination flag indicating whether the transition leads to a terminal state/limit
        next_state : torch.Tensor
            Tensor representation of the next state
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (
            state,
            action,
            reward,
            done,
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

    def state_dict(self) -> dict:
        """
        Return a dictionnary of a buffer instance

        Returns
        -------
        dict
            The instance
        """
        return {
            "capacity": self.capacity,
            "memory": self.memory,
            "position": self.position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load a buffer from a dictionnary

        Parameters
        ----------
        state_dict : dict
            The dictionnary used for the loading
        """
        self.capacity = int(state_dict["capacity"])
        self.memory = state_dict["memory"]
        self.position = int(state_dict["position"])
