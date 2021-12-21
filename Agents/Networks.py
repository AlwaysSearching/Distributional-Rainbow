from torch import nn
from torch import optim
import torch

import numpy as np

from Agents.Layers import NoisyLinear

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, act_dim, n_hid=128, noisy=True, clip_grad_val=None):
        super(DuelingNetwork, self).__init__()
        
        if isinstance(state_dim, (list, tuple)):
            state_dim = np.prod(state_dim)
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()
        
        self.input_layer = nn.Linear(state_dim, n_hid)
        self.hidden_layer_1 = nn.Linear(n_hid, n_hid)
        self.hidden_layer_2 = nn.Linear(n_hid, n_hid) 
        self.hidden_layer_3 = NoisyLinear(n_hid, n_hid) if noisy else nn.Linear(n_hid, 1)
        
        self.state_value_layer = NoisyLinear(n_hid, 1) if noisy else nn.Linear(n_hid, 1)
        self.action_value_layer = NoisyLinear(n_hid, act_dim) if noisy else nn.Linear(n_hid, act_dim)
        
        self.noisy_layers = ['state_value_layer', 'action_value_layer', 'hidden_layer_3']
        
        self.ReLU = nn.ReLU()
        self.clip_grad_val = clip_grad_val
        
    def forward(self, states):
        states = self.flatten(states)
        
        x = self.ReLU(self.input_layer(states))        
        x = self.ReLU(self.hidden_layer_1(x))
        x = self.ReLU(self.hidden_layer_2(x))
        
        state_value = self.state_value_layer(x)
        action_value = self.action_value_layer(x)
        
        # Q(s, a) = V(s) + A(s, a) - sum_{a \in A}{A(s, a)} / |A|
        return state_value + action_value - torch.mean(action_value, dim=-1, keepdims=True)
    
    def train_step(self, loss, optim, lr_scheduler=None):
        optim.zero_grad()
        loss.backward()
        
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)        
        optim.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        return loss
    
    def reset_noise(self):
        for name, module in self.named_children():
            if name in self.noisy_layers:
                module.reset_noise()
    
    def polyak_update(self, source_network, source_ratio=0.5):
        '''
        Update the parameters of the network with the parameters of the source network. 
        source_ratio = 1.0 simply performs a copy, rather than a polyak update.
        '''
        
        for src_param, param in zip(source_network.parameters(), self.parameters()):
            param.data.copy_(source_ratio*src_param.data + (1.0 - source_ratio)*param.data)