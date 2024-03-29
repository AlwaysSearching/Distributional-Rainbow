import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NoisyLinear(nn.Module):
    '''
    Noisy Linear Layer for exploration. Factorized version used for parameter efficiency.

    Based on the following paper:
        Noisy Networks for Exploration, 2018
        https://arxiv.org/pdf/1706.10295.pdf
    '''

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        """
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        std_init : float
            Initial standard deviation of the weights.
        """
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register buffers will not be optimized by the optimizer but will be saved in the state dict
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):

        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
