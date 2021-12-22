from torch import nn
from torch import optim
import torch

import numpy as np

from Agents.Layers import NoisyLinear

class DuelingNetwork(nn.Module):
    def __init__(self, frame_hist, act_dim, atoms=1, n_hid=128, noisy=True, clip_grad_val=None, atoms=None):
        super(DuelingNetwork, self).__init__()
        # Atom = 1 allows us to use the below for standard RL. 
        self.atoms = atoms
        self.hist = frame_hist

        self.conv_layers = nn.Sequential(
                nn.Conv2d(self.hist, 32, 5,  stride=5, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU()
        )
        self.conv_output_dim = 576

        self.fc_hidden_layer_1 = nn.Linear(self.conv_output_dim, n_hid)
        
        self.state_value_layer_1 = NoisyLinear(n_hid, n_hid) if noisy else nn.Linear(n_hid, 1)
        self.state_value_layer_2 = NoisyLinear(n_hid, self.atoms) if noisy else nn.Linear(n_hid, 1) 

        self.advantage_layer_1 = NoisyLinear(n_hid, n_hid) if noisy else nn.Linear(n_hid, act_dim)
        self.advantage_layer_2 = NoisyLinear(n_hid, self.atoms * act_dim) if noisy else nn.Linear(n_hid, act_dim)
        
        self.noisy_layers = ['state_value_layer_1', 'state_value_layer', 'action_value_layer_1', 'action_value_layer_2']
        
        self.ReLU = nn.ReLU()
        self.clip_grad_val = clip_grad_val
        
    def forward(self, states):
        # pass image through conv layers and flatten the output
        x = self.conv_layers(x).view(-1, self.conv_output_dim) 

        # single feed_forwad layer prior to value and advantage streams
        x = self.ReLU(self.fc_hidden_layer_1(x))
        
        #  Value Stream
        state_value = self.Relu(self.state_value_layer_1(x))
        state_value = self.state_value_layer_2(state_value)

        # Advantage Stream
        advantage = self.Relu(self.advantage_layer_1(x))
        advantage = self.advantage_layer_2(advantage)
        
        # Q(s, a) = V(s) + A(s, a) - sum_{a \in A}{A(s, a)} / |A|
        action_value = state_value + advantage - torch.mean(advantage, dim=-1, keepdims=True)
        return action_value
    
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
        # resample the noise parameter for each Noisy layer in the network
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
