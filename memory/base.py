from typing import TypedDict
from abc import ABC, abstractmethod

import torch


class ReplayBatch(TypedDict):
    """A batch of data returned by a replay buffer."""
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class Memory(ABC):
    def __init__(self):
        self.data_keys = ['states', 'actions', 'next_states', 'rewards', 'dones', 'timestep']

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        '''Implement memory update given the full info from the latest timestep.'''
        pass

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        pass


