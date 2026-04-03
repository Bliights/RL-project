import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int) -> None:
        """
        Initialise the feedforward neural network used to approximate the Q-function

        Parameters
        ----------
        obs_dim : int
            Dimension of the flattened observation space
        hidden_size : int
            Size of the hidden fully connected layer
        n_actions : int
            Number of possible discrete actions
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of input states

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Tensor containing one predicted Q-value per action
        """
        return self.network(x)
