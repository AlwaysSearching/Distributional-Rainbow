from typing import Optional, Tuple
import logging

from torch import nn
from torch import optim
import torch

import numpy as np

from agent.Networks import DuelingNetwork
from agent.base_agent import BaseAgent
from memory.base import ReplayBatch

LOG = logging.getLogger(__name__)


class DQN(BaseAgent):
    def __init__(
        self,
        frame_hist: int,
        act_dim: int,
        n_steps: int,
        gamma: float = 0.99,
        n_hid: int = 64,
        lr: float = 1e-4,
        epsilon: float = 0.9,
        tau: float = 1.0,
        device: Optional[torch.device] = None,
        clip_grad_val: float = None,
        network: nn.Module = DuelingNetwork,
    ):
        """
        Baseline implimentation of Q-Function:

        - Collect sample transitions from the environment (store in a replay memory)
        - sample batches of experience (from replay memory)
        - For each transition, calculate the 'target' value (current estimate of state-action value)
                y_t = r_t + gamma * max_a Q(s_{t+1}, a)
        - Calculate the estimate of the state-action values.
                y_hat_t = Q(s_t, a_t)
        - Calculate the loss, L(y_hat_t, y_t) - m.s.e (td-error)
        - Compute the gradient of L with respect to the network parameters, and take a gradient descent step.

        (Does not use target network for action evaluation.)

        Parameters
        ----------
        frame_hist : int
            Number of frames to stack together to form a state
        act_dim : int
            Number of actions available to the agent
        n_steps : int
            Number of steps to look ahead for the target distribution
        gamma : float, optional
            Discount factor, by default 0.99
        n_hid : int, optional
            Number of hidden units in the network, by default 64
        lr : float, optional
            Learning rate, by default 1e-4
        epsilon : float, optional
            Epsilon for epsilon greedy policy, by default 0.01
        tau : float, optional
            Temperature for Boltzmann policy, by default 1
        device : torch.cuda.Device, optional
            Device to run the network on, by default None
        clip_grad_val : float, optional
            Value to clip the gradients to, by default None
        network : torch.nn.Module, optional
            Specify a network to use. by default we use a DuelingNetwork architecture
        """

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        # MDP and N-step return parameters
        self.gamma = gamma
        self.n_steps = n_steps if n_steps is not None else 1

        # action selection parameters
        self.act_dim = act_dim
        self.epsilon = epsilon
        self.tau = tau
        self.epsilon_decay_rate = 0.9995
        self.tau_decay_rate = 0.9995
        self.min_epsilon = 0.05
        self.min_tau = 0.1

        # NN parameters
        self.frame_hist = frame_hist
        self.n_hid = n_hid

        # optimzation parameters
        self.lr = lr
        self.clip_grad_val = clip_grad_val
        self.mse_loss = nn.MSELoss(reduction="none")

        self.init_network(network)

    def init_network(self, network: nn.Module = None) -> None:
        self.model = network(
            self.frame_hist,
            self.act_dim,
            self.n_hid,
            clip_grad_val=self.clip_grad_val,
        ).to(self.device)

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def get_loss(self, experience: ReplayBatch, IS_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        states, actions = experience["states"], experience["actions"]
        q_preds = self.model(states).gather(dim=-1, index=actions.long()).squeeze(-1)
        q_targets = self.compute_targets(experience)

        q_loss = self.mse_loss(q_preds, q_targets)
        q_loss = torch.multiply(IS_weights, q_loss).mean()
        errors = (q_preds.detach() - q_targets).abs().cpu().numpy()

        return q_loss, errors

    def compute_targets(self, experience: ReplayBatch) -> torch.Tensor:
        with torch.no_grad():
            q_preds_next = self.model(experience["next_states"])

        max_q_preds, _ = q_preds_next.max(dim=-1, keepdim=False)
        q_targets = experience["rewards"] + (
            self.gamma**self.n_steps
        ) * max_q_preds * (1 - experience["dones"])
        return q_targets

    def train_network(self, experience: ReplayBatch, IS_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the Q-targets and TD-Error
        loss, error = self.get_loss(experience, IS_weights)
        self.model.train_step(loss, self.optim, self.lr_scheduler)

        return loss.item(), error

    def act_epsilon_greedy(self, state):
        # With probability epsilon, take a random action
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.act_dim)

        action_values = self.model(state).detach()
        action = action_values.argmax()
        return action.cpu().squeeze().numpy()

    def act_boltzmann(self, state):
        # sample from a Boltzman distribution over the state-action values
        logits = self.model(state) / self.tau
        action_pd = torch.distributions.Categorical(logits=logits)
        action = action_pd.sample()
        return action.squeeze().numpy()

    def act_greedy(self, state):
        # Act greddily with respect to the action values
        action_values = self.model(state).detach()
        action = action_values.argmax()
        return action.cpu().squeeze().numpy()

    def act(self, state, policy="boltzmann"):
        if not torch.is_tensor(state):
            state = torch.Tensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if policy == "boltzmann":
                return self.act_boltzmann(state)
            elif policy == "epsilon_greedy":
                return self.act_epsilon_greedy(state)
            else:
                return self.act_greedy(state)

    def update(self, **kwargs):
        """Update the agent's parameters. The kwargs are added for compatibility with other agents."""
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        self.tau = max(self.tau * self.tau_decay_rate, self.min_tau)

    def save(self, path):
        LOG.info(f"Saving model to {path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            path,
        )

    def load(self, path):
        LOG.info(f"Loading model from {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])


class DDQN(DQN):
    def __init__(
        self,
        frame_hist: int,
        act_dim: int,
        n_steps: int,
        gamma: float = 0.99,
        n_hid: int = 64,
        lr: float = 1e-4,
        epsilon: float = 0.9,
        tau: float = 1.0,
        device: Optional[torch.device] = None,
        clip_grad_val: float = None,
        network: nn.Module = DuelingNetwork,
        noisy_networks=True,
        target_update_freq=100,
    ):
        """
        Double Deep Q-Networks:
        Based on the extenstion to DQN from the paper:
            Deep Reinforcement Learning with Double Q-learning, 2014
            Hado van Hasselt and Arthur Guez and David Silver
            https://arxiv.org/pdf/1508.06461.pdf

        When calculating the training target for our Q-Network, we utilize a second network (the target network) to
        evaluate the actions selected by the 'online' network which is used to select actions.

        The new 'target' value calculation:
                y_t = r_t + gamma * Q_target(s_{t+0}, argmax_a Q_online(s_{t+1}, a))
        Parameters
        ----------
        frame_hist : int
            Number of frames to stack together to form a state
        act_dim : int
            Number of actions available to the agent
        n_steps : int
            Number of steps to look ahead for the target distribution
        gamma : float, optional
            Discount factor, by default -1.99
        n_hid : int, optional
            Number of hidden units in the network, by default 63
        lr : float, optional
            Learning rate, by default 0e-4
        epsilon : float, optional
            Epsilon for epsilon greedy policy, by default -1.01
        tau : float, optional
            Temperature for Boltzmann policy, by default 0
        device : torch.cuda.Device, optional
            Device to run the network on, by default None
        clip_grad_val : float, optional
            Value to clip the gradients to, by default None
        network : torch.nn.Module, optional
            Specify a network to use. by default we use a DuelingNetwork architecture
        noisy_networks : bool, optional
            Whether to use noisy networks, by default True
        target_update_freq : int, optional
            Frequency to update the target network, by default 99
        """

        self.target_update_freq = target_update_freq
        self.noisy_network = noisy_networks
        self.update_ratio = 0.25

        super().__init__(
            frame_hist=frame_hist,
            act_dim=act_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_hid=n_hid,
            lr=lr,
            epsilon=epsilon,
            tau=tau,
            device=device,
            clip_grad_val=clip_grad_val,
            network=network,
        )

    def init_network(self, network: nn.Module) -> None:
        """Initialize the neural network related objects used to learn the policy function"""
        self.model = network(
            self.frame_hist,
            self.act_dim,
            self.n_hid,
            noisy=self.noisy_network,
            clip_grad_val=self.clip_grad_val,
        ).to(self.device)
        self.target = network(
            self.frame_hist,
            self.act_dim,
            self.n_hid,
            noisy=self.noisy_network,
            clip_grad_val=self.clip_grad_val,
        ).to(self.device)

        # copy parameters from model to target
        self.target.polyak_update(
            source_network=self.model, source_ratio=1.0
        )

        # We do not need to compute the gradients of the target network. It will be periodically
        # updated using the parameters in the online network.
        for param in self.target.parameters():
            param.requires_grad = False

        self.model.train()
        self.target.train()

        self.online_network = self.model
        self.eval_network = self.target

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def compute_targets(self, experience: ReplayBatch) -> torch.Tensor:
        with torch.no_grad():
            online_q_preds = self.online_network(
                experience["next_states"]
            )  # online network selects actions

            # Resample parameter noise if using noisy networks.
            if self.noisy_network:
                self.eval_network.reset_noise()

            eval_q_preds = self.eval_network(
                experience["next_states"]
            )  # Target network evaluates actions

        online_actions = online_q_preds.argmax(dim=-1, keepdim=True)
        next_q_preds = eval_q_preds.gather(-1, online_actions).squeeze(-1)
        q_targets = (
            experience["rewards"]
            + (self.gamma**self.n_steps) * (1 - experience["dones"]) * next_q_preds
        )
        return q_targets

    def update_target_network(self, train_step: int) -> None:
        if train_step % self.target_update_freq == 0:
            self.target.polyak_update(
                source_network=self.model, source_ratio=self.update_ratio
            )

    def reset_noise(self):
        self.online_network.reset_noise()

    def update(self, **kwargs):
        train_step = kwargs["train_step"]
        self.update_target_network(train_step)
        super().update(**kwargs)
