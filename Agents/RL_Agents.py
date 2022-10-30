from typing import Literal

from Agents.memory.experience_replay import PrioritizedExperienceReplay
from Agents.QNetworks import DQN, DDQN
from Agents.dist_QNetworks import Categorical_DDQN

POLICY = {
    "DQN": DQN,
    "DDQN": DDQN,
    "Categorical_DDQN": Categorical_DDQN,
}


class RL_Agent:
    def __init__(
        self,
        agent_name: list[Literal["DQN", "DDQN", "Categorical_DDQN"]],
        state_dim,
        action_dim,
        gamma=0.99,
        lr=5e-4,
        tau=3,
        batchsize=16,
        memory_min_train_size=256,
        memory_maxlength=5000,
        train_freq=5,
        noisy_networks=True,
        n_step_return=None,
        device=None,
    ):
        """
        Parameters
        ----------
        agent_name : list[Literal["DQN", "DDQN", "Categorical_DDQN"]]
            Specify the agent to used.
        state_dim : int
            Dimension of the state/observation space
        action_dim : int
            Dimension of the action space. For DQN & DDQN this must be an integer value.
        gamma : float
            Discount Factor - default = 0.99
        batchsize : int
            Training batch Size - default = 16
        lr : float
            Learning Rate for the Q-Network - default 5e-4
        tau : float
            Temperature parameter for softmax action selection
        memory_min_train_size : int
            Number of transitions to observe before first policy train step. default = 64
        memory_maxlength : int
            Maximum number of experiences to hold in the memory buffer. default = 1000
        train_freq : int
            Frequency of policy updates. Train Policy every train_freq environment steps. default = 5
        noisy_networks : bool
            Whether to use Noisy Layers in the DDQN. default = True
        n_step_return : int
            Number of steps to use in computing N-step returns. default to None (1-step return)
        device : torch.device
            Pass a device i.e. gpu/cuda/cpu to be used by the agent and replay buffer
        """

        self.memory = PrioritizedExperienceReplay(
            state_dim=state_dim,
            action_dim=0,
            min_train_size=memory_min_train_size,
            max_size=memory_maxlength,
            n_step_return=n_step_return,
            gamma=gamma,
            device=device,
        )

        self.policy = POLICY[agent_name](
            frame_hist=3,
            state_dims=state_dim,
            action_dim=action_dim,
            n_step_return=n_step_return,
            lr=lr,
            tau=tau,
            gamma=gamma,
            noisy_networks=noisy_networks,
            device=device,
        )

        self.memory.PER_b_increment = 0.000001
        self.batchsize = batchsize
        self.epochs = 4  # how many training iterations per update
        self.count = 0
        self.train_freq = train_freq
        self.noisy_networks = noisy_networks

    def act(self, state, policy=None):
        if policy is None:
            policy = "epsilon_greedy" if self.noisy_networks else "boltzmann"
        return self.policy.act(state, policy=policy).tolist()

    def update(self, state, action, reward, next_state, done):
        self.memory.update(state, [action], reward, next_state, int(done))
        self.count += 1

        if done:
            self.policy.update()

        # Sample parameter noise if using Noisy Networks
        if self.noisy_networks:
            self.policy.reset_noise()

        if self.memory.min_train_size_reached() and self.count % self.train_freq == 0:
            avg_loss = 0.0

            for _ in range(self.epochs):
                # Sample transitions from the replay memory and train the policy network
                tree_idxs, batch, IS_weights = self.memory.sample(self.batchsize)
                loss, error = self.policy.train_network(batch, IS_weights)

                # update the priorities of the sampled transitions.
                self.memory.update_priorities(tree_idxs, error)
                avg_loss += loss

            self.policy.update_target_network(self.count // self.train_freq)
            return avg_loss / self.batchsize
