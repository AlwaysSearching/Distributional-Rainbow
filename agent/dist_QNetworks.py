from typing import Optional, Tuple

from torch import nn
from torch import optim
import torch

from Networks import DuelingNetwork
from QNetworks import DDQN


class Categorical_DDQN(DDQN):
    def __init__(
        self,
        frame_hist: int,
        act_dim: int,
        atoms: int,
        V_min: float,
        V_max: float,
        n_steps: int,
        gamma: float = 0.99,
        n_hid: int = 64,
        lr: float = 1e-4,
        epsilon: float = 0.01,
        device: Optional[torch.cuda.Device] = None,
        clip_grad_val: Optional[float] = None,
        network: Optional[nn.Module] = None,
    ):
        """
        Implimentation of the Categorical DQN introduced in the paper:
            A Distributional Perspective on Reinforcement Learning, 2017
            https://arxiv.org/pdf/1707.06887.pdf

        Parameters
        ----------
        frame_hist : int
            Number of frames to stack together to form a state
        act_dim : int
            Number of actions available to the agent
        atoms : int
            Number of atoms in the support of the value distribution
        V_min : float
            Minimum value of the support
        V_max : float
            Maximum value of the support
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
        device : torch.cuda.Device, optional
            Device to run the network on, by default None
        """
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        # Categorical Distribution hyper params
        self.atoms = atoms
        self.V_min = V_min
        self.V_max = V_max
        self.support = torch.linspace(V_min, V_max, atoms).to(self.device)
        self.dz = (V_max - V_min) / (atoms - 1)

        # NN hyper parameters
        self.act_dim = act_dim
        self.frame_hist = frame_hist
        self.gamma = gamma
        self.n_hid = n_hid
        self.lr = lr

        # action_selection hyper parameters
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.9999
        self.min_epsilon = 0.001

        self.clip_grad_val = clip_grad_val
        self.n_steps = n_steps
        self.init_network(network)

    def init_network(self, network: nn.Module = None) -> None:
        """Initialize the network and optimizer"""
        if network is None:
            self.model = DuelingNetwork(
                self.frame_hist,
                self.act_dim,
                atoms=self.atoms,
                n_hid=self.n_hid,
                clip_grad_val=self.clip_grad_val,
            ).to(self.device)
            self.target = DuelingNetwork(
                self.frame_hist,
                self.act_dim,
                atoms=self.atoms,
                n_hid=self.n_hid,
                clip_grad_val=self.clip_grad_val,
            ).to(self.device)
        else:
            self.model = network(
                self.frame_hist,
                self.act_dim,
                atoms=self.atoms,
                n_hid=self.n_hid,
                clip_grad_val=self.clip_grad_val,
            ).to(self.device)
            self.target = network(
                self.frame_hist,
                self.act_dim,
                atoms=self.atoms,
                n_hid=self.n_hid,
                clip_grad_val=self.clip_grad_val,
            ).to(self.device)

        self.target.polyak_update(
            source_network=self.model, source_ratio=1.0
        )  # copy parameters from model to target

        self.model.train()
        self.target.train()

        self.online_network = self.model
        self.eval_network = self.target

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def train_network(
        self,
        experience: dict[str, torch.Tensor],
        IS_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute the Cross Entropy loss using the distribution of rewards and subsequent states
        loss, error = self.get_loss(experience, IS_weights)

        # Apply importance sampling weights to the loss and train the Q-Value function
        loss = (IS_weights * loss).mean()
        self.model.train_step(loss, self.optim, self.lr_scheduler)

        return loss.item(), error

    def get_loss(self, experience: dict[str, torch.Tensor]) -> torch.Tensor:
        # Log probabilities of action-value dist. of the online networks Used for Cross Entropy Loss
        log_ps_x = nn.functional.log_softmax(
            self.online_network(experience["states"]), dim=2
        )  # log(p(x_t, ·; θ_online))
        log_ps_x_a = log_ps_x[:, experience["actions"]]  # log(p(x_t, a_t; θ_online))

        # compute the target distributions using next states and returns
        m = self.compute_targets(experience)
        # minimize distance between Z_θ and TZ_θ -> Cros entropy Loss
        loss = -torch.sum(m * log_ps_x_a, 1)

        return loss, loss.detach().abs().cpu().numpy()

    def compute_targets(self, experience: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the target distribution for the next states using the Distributional Bellman Equation:
            1. Select the optimal action with respect to the online network
            2. Compute the Q-value distribution for the next state using the target network and the action
               chosen by the online network.
            3. Apply the distributional Bellman operator using the observed rewards and project onto the
               support of the Q-value distribution
            4. Distribute the probability mass of the projected distribution onto the support of the Q-value
               distribution
        """
        states = experience["states"]
        actions = experience["actions"]
        next_states = experience["next_states"]
        rewards = experience["rewards"]
        is_terminal = experience["dones"]

        batch_size = states.shape[0]
        with torch.no_grad():

            # Select the argmax action using the online networks
            pr_x_online = nn.functioanal.softmax(self.online_network(next_states))  # p(s_t+n, ·; θ_online)
            value_dist_online = self.support.expand_as(pr_x_online) * pr_x_online   # z * p(s_t+n, ·; θ_online)
            argmax_idxs = value_dist_online.sum(2).argmax(1)  # argmax_{a} sum_{j} z_j p_{j}(s_t+n, a; θ_online)

            # Compute the Q-Value probablities of the argmax action using the target network
            self.eval_network.reset_noise()
            pr_x_target = torch.nn.functioanal.softmax(self.eval_network(next_states))  # p(s_t+n, ·; θ_target)
            pr_x_a_target = pr_x_target[:, argmax_idxs]  # p(s_t+n, a*; θ_target)

            # Compute Tz and project onto the chosen support. (T is the Bellman operator for distributions)
            Tz = (
                rewards.unsqueeze(1) + is_terminal * (self.gamma**self.n_steps) * self.support.unsqueeze(0)
            ).clamp(self.V_min, self.V_max)

            # projection onto the fixed support for z
            b = (Tz - self.V_min) / self.dz
            lower = b.floor().to(torch.int64)
            upper = b.ceil().to(torch.int64)

            # disappearing probability mass when l = b = u
            lower[(upper > 0) * (lower == upper)] -= 1  # if b == V_min
            upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1  # if b == V_max

            m = torch.zeros(batch_size * self.atoms).to(self.device).float()
            off_set = (
                torch.linspace(0, (batch_size - 1) * self.atoms, batch_size)
                .unsqueeze(0)
                .expand(batch_size, self.atoms)
                .to(actions)
            )  # line up the value of Tz with the distribution of atoms.

            # Distribute the probabilities to each atom of Tz
            m.index_add(
                0, (lower + off_set).view(-1), (pr_x_a_target * (upper - b)).view(-1)
            )  # m_l += p(s_{t+n}, a*) * (u - b)
            m.index_add(
                0, (upper + off_set).view(-1), (pr_x_a_target * (b - lower)).view(-1)
            )  # m_lu+= p(s_{t+n}, a*) * (b - l)
            m = torch.reshape(m, [batch_size, self.atoms])

        return m

    def act_greedy(self, state: torch.Tensor) -> int:
        # advantage distribution for each action
        action_prob = nn.functional.softmax(self.model(state).detach(), dim=2)
        # Compute Expected action values based on the value distributions
        expected_action_values = (action_prob * self.support).sum(2)

        return expected_action_values.argmax(1).item()
