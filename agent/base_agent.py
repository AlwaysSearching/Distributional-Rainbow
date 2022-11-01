from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from memory.base import ReplayBatch


class BaseAgent(ABC, nn.Module):
    """
    Define the general Agent API so that we can exchange agents with out
    needing to rewrite the training/testing code.
    """

    @abstractmethod
    def act(self, state: torch.Tensor, policy: str) -> np.ndarray:
        """
        Given a state, return an action and the log probability of the action
        """
        pass

    @abstractmethod
    def init_network(self, network: nn.Module = None) -> None:
        """Initialize the network and optimizer"""
        pass

    @abstractmethod
    def train_network(self, experience: ReplayBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performe a single trainint step for the online network"""
        pass

    @abstractmethod
    def get_loss(self, experience: ReplayBatch, IS_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the loss for the batch"""
        pass

    @abstractmethod
    def compute_target(self, experience: ReplayBatch) -> torch.Tensor:
        """Compute the target for the batch"""
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the agent to the given path.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the agent from the given path.
        """
        pass

    def update_target_network(self):
        """
        Update the target network using the online network weights
        """
        pass

    def reset_noise(self):
        """
        Reset the noise in the Noisy Net Layers of the agent.
        """
        pass

    def update(self, **kwargs):
        """
        update the network hyperparameters or target networks (i.e. epsilon/tau for exploration)
        """
        pass
