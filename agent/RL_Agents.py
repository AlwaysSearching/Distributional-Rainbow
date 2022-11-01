from typing import Literal, Union

from memory.experience_replay import PrioritizedExperienceReplay, ReplayBuffer
from agent.QNetworks import DQN, DDQN
from agent.dist_QNetworks import Categorical_DDQN

from configs import AgentConfig

POLICY = {
    "DQN": DQN,
    "DDQN": DDQN,
    "Categorical_DDQN": Categorical_DDQN,
}

REPLAY_BUFFER = {
    "PER": PrioritizedExperienceReplay,
    "Replay": ReplayBuffer,
}


class RL_Agent:
    def __init__(
        self,
        config: AgentConfig,
        device=None,
    ):
        """
        Parameters
        ----------
        config : AgentConfig
            Agent Configuration and training parameters.
        device : torch.device
            Pass a device i.e. gpu/cuda/cpu to be used by the agent and replay buffer
        """

        self.memory = REPLAY_BUFFER[config.buffer_type](
            config,
            device=device,
        )

        self.policy = POLICY[config.agent_name](
            config,
            device=device,
        )

        self.batchsize = config.batchsize
        self.train_freq = config.train_freq
        self.noisy_networks = config.noisy_networks

        # how many training iterations per update
        self.epochs = config.iterations_per_epoch
        self.count = 0

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

            self.policy.update_target_network(self.count)
            return avg_loss / self.batchsize
