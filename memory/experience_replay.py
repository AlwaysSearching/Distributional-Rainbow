from typing import Optional, Tuple, Union

from memory.base import Memory, ReplayBatch
from memory.sumtree import SumTree
from configs import AgentConfig

import numpy as np
import torch


class ReplayBuffer(Memory):
    """
    Stores agent experiences and samples from them for agent training
    An experience consists of
     - state: representation of a state
     - action: action taken
     - next state: representation of next state (should be same as state)

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
     - This is implemented as a circular buffer so that inserting experiences are O(1)
     - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    NOTE: This implimentation does not distinguish between epsiode truncation and episode termination. This is not an issue
    if the environment does truncate episodes after a certain number of steps, but may lead to unexpected behavior otherwise.
    """

    def __init__(
        self,
        config: AgentConfig,
        device: torch.device = None
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.min_train_size = config.min_train_size
        self.max_size = config.max_size

        self.size = 0           # total experiences stored
        self.seen_size = 0      # total experiences seen cumulatively
        self.head = -1          # index of most recent experience
        self.t = 0              # timestep of most recent experience

        self.state_dim = config.state_dim if isinstance(config.state_dim, (list, tuple)) else [config.state_dim]
        self.action_dim = config.action_dim if isinstance(config.action_dim, (list, tuple)) else [config.action_dim]

        # declare what data keys to store
        self.reset()

        self.n_step = 1 if config.n_step_return is None else config.n_step_return
        self.gamma = config.gamma
        self.n_step_gamma = torch.pow(self.gamma, torch.arange(self.n_step)).to(torch.float32).to(self.device)

    def reset(self):
        """Initialize the memory Tensors, size, and head pointer"""

        setattr(self, "states", torch.zeros([self.max_size, *self.state_dim], dtype=torch.float32).to(self.device))
        setattr(self, "actions", torch.zeros([self.max_size, *self.action_dim], dtype=torch.int16).to(self.device))
        setattr(self, "next_states", torch.zeros([self.max_size, *self.state_dim], dtype=torch.float32).to(self.device))
        setattr(self, "rewards", torch.zeros([self.max_size], dtype=torch.float32).to(self.device))
        setattr(self, "dones", torch.zeros([self.max_size], dtype=torch.int8).to(self.device))
        setattr(self, "timestep", torch.zeros([self.max_size], dtype=torch.int32).to(self.device))

        self.size = 0
        self.head = -1

    def update(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: Union[np.ndarray, torch.Tensor],
        next_state: Union[np.ndarray, torch.Tensor],
        done: Union[np.ndarray, torch.Tensor],
        truncated: bool = False
    ) -> None:
        """Implementation for update() to add experience to memory, expanding the memory size if necessary"""

        # Move head pointer. Wrap around if necessary.
        self.head = (self.head + 1) % self.max_size

        self.states[self.head] = torch.tensor(state).to(self.device)
        self.actions[self.head] = action if isinstance(action, torch.Tensor) else torch.tensor(action).to(self.device)
        self.rewards[self.head] = reward
        self.next_states[self.head] = torch.tensor(next_state).to(self.device)
        self.dones[self.head] = done
        self.timestep[self.head] = self.t

        # size of occupied memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1

        self.t = 0 if done else self.t + 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ReplayBatch, torch.Tensor]:
        """
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience.
        Values are tensors of the corresponding elements

        batch = {
            "states"     : states,
            "actions"    : actions,
            "next_states": next_states,
            "rewards"    : rewards,
            "dones"      : dones
        }
        """

        batch_idx = self.sample_idxs(batch_size)
        return batch_idx, self._get(batch_idx), torch.ones(batch_size).to(self.device)

    def _get(self, batch_idx: np.ndarray) -> ReplayBatch:
        """Returns a batch of experiences given a list of indices"""
        # if we are using N-step returns, extend the indxs for n steps.
        if self.n_step is not None:
            batch_idx = torch.stack(
                [torch.arange(idx, idx + self.n_step, dtype=torch.int64) % self.max_size for idx in batch_idx]
            )

        batch = ReplayBatch()
        for key in self.data_keys:
            batch[key] = getattr(self, key)[batch_idx]

        # handle N-step returns
        if self.n_step is not None:
            # We need to check if the episode terminates within the following n-steps
            transitions_firsts = batch["timestep"] == 0
            mask = torch.zeros_like(transitions_firsts).to(self.device)

            # check if there is a terminal state at or prior to the current state.
            for t in range(1, self.n_step):
                mask[:, t] = torch.logical_or(mask[:, t - 1], transitions_firsts[:, t])

            # use the mask to zero out rewards after the terminal step,
            # and to set the done values to 1 for all steps after the terminal state
            batch["rewards"] = (1 - mask.to(torch.int32))*batch["rewards"]
            batch["dones"] = torch.logical_or(mask[:, -1], batch["dones"][:, -1]).to(torch.float16)

            # compute sum of discounted rewards
            batch["rewards"] = torch.sum(torch.mul(self.n_step_gamma, batch["rewards"]), axis=1)

            # take the initial state, the action, and the final state (this is can be any value if done == 1)
            batch["states"] = batch["states"][:, 0]
            batch["actions"] = batch["actions"][:, 0]
            batch["next_states"] = batch["next_states"][:, -1]

        return batch

    def sample_idxs(self, batch_size: int) -> np.ndarray:
        """Batch indices a sampled random uniformly"""

        valid = False
        while not valid:
            batch_idx = np.random.choice(
                self.size, size=batch_size, replace=False, p=[1 / self.size for i in range(self.size)]
            )
            # In the case of N-step returns, we need to ensure that the sampled indices will not cross the head value.
            # i.e. current head is 3 -> an n-step transition with indices (2, 3, 4) is invalid
            if self.n_step is None or np.all((self.head - batch_idx) % self.max_size > self.n_step):
                valid = True

        # if buffer is full, wrap the indices around to the beginning of the buffer
        if self.size == self.max_size:
            batch_idx = (self.head + batch_idx) % self.size

        return batch_idx

    def min_train_size_reached(self):
        return self.seen_size >= self.min_train_size

    def update_n_step(self, new_n_step: int, new_gamma: float = None) -> None:
        """Update the length of the n-step returns to use in sampling."""
        self.n_step = new_n_step
        self.gamma = self.gamma if new_gamma is None else new_gamma
        self.n_step_gamma = torch.pow(self.gamma, torch.arange(self.n_step)).to(torch.float32).to(self.device)

    def update_priorities(self, tree_idx: np.ndarray, abs_errors: torch.Tensor) -> None:
        """
        When using standard replay buffer there is no priority sampling mechanism.
        This is to simplify interface between memory buffer and agents.
        """
        pass


class PrioritizedExperienceReplay(Memory):
    """
    Impliment a Replay buffer which utilizes the prioritized replay mechanism from
        PRIORITIZED EXPERIENCE REPLAY, 2016
        Tom Schaul, John Quan, Ioannis Antonoglou and David Silver
        https://arxiv.org/pdf/1511.05952.pdf

    Data is stored on GPU, and indexes are produced via a sum tree which tracks the priorities of each transition.
    """
    def __init__(
        self,
        config: AgentConfig,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.ReplayBuffer = ReplayBuffer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            min_train_size=config.min_train_size,
            max_size=config.max_size,
            n_step_return=config.n_step_return,
            gamma=config.gamma,
            device=device
        )
        self.SumTree = SumTree(config.max_size)

        # Tradeoff between sampling only high priority experiences and sampling randomly
        self.PER_a = config.PER_a if config.PER_a else 0.6
        # Address the induced samplming bias via importance-sampling weights
        self.PER_b = config.PER_b if config.PER_b else 0.4
        # Ensure all experiences have a non-zero probability of being samples
        self.PER_e = config.PER_e if config.PER_e else 0.01

        self.PER_b_increment = config.PER_b_increment if config.PER_b_increment else 0.000001
        self.max_absolute_error = config.PER_max_priority if config.PER_max_priority else 30.0

    def reset(self) -> None:
        self.ReplayBuffer.reset()
        self.SumTree.reset()

    def update(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: Union[np.ndarray, torch.Tensor],
        next_state: Union[np.ndarray, torch.Tensor],
        done: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Add new transition to the replay buffer.
        Each new experience is given max_prority to ensure that it is trained on
        """

        # Find the max priority
        max_priority = np.max(self.SumTree.tree[-self.SumTree.capacity:])

        # If the max priority = 0, use a minimum priority to ensure it is trained on.
        if max_priority == 0:
            max_priority = self.max_absolute_error

        # Add priority to the replay buffer.
        self.SumTree.add(max_priority)

        # Add experience to the Replay Buffer
        self.ReplayBuffer.update(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ReplayBatch, torch.Tensor]:
        """
        Sample a minibatch of k size.
        First split the tree into k ranges. Then sample values uniformly from each range.
        Using the sumtree, the experience whose priority score corresponds to sampled values are retrieved.
        Finally, we calculate the Importnace Sampling weights for each transition.
        """

        # indexes in the priority tree, the replay buffer, and the importance sampling weights.
        tree_idxs = np.empty((batch_size,), dtype=np.int32)
        batch_idxs = np.empty((batch_size,), dtype=np.int32)
        IS_Weights = np.empty((batch_size,), dtype=np.float32)

        # divide the tree into n ranges
        priority_segment = self.SumTree.total_priority / batch_size  # priority segment

        # Increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment])  # max = 1

        # Calculating the max_weight
        capacity = self.SumTree.capacity
        leafs = self.SumTree.tree[-capacity:]
        total_priority = self.SumTree.total_priority

        min_priority = np.min(leafs[np.nonzero(leafs)]) / total_priority
        max_weight = (min_priority * batch_size) ** (-self.PER_b)

        for k in range(batch_size):
            a, b = priority_segment*k, priority_segment*(k + 1)
            v = np.random.uniform(a, b)
            tree_index, replay_buffer_index, priority = self.SumTree.get_leaf(v)
            sampling_probabilities = priority / self.SumTree.total_priority

            #  IS = (1/N * 1/P(i))**b / (max wi) == (N*P(i))**-b  / (max wi)
            IS_Weights[k] = np.power(batch_size * sampling_probabilities, -self.PER_b) / max_weight

            # add the tree index and the replay buffer index.
            tree_idxs[k] = tree_index
            batch_idxs[k] = replay_buffer_index

        batch = self.ReplayBuffer._get(batch_idxs)
        return tree_idxs, batch, torch.tensor(IS_Weights).to(self.ReplayBuffer.device)

    def update_priorities(self, tree_idx: np.ndarray, abs_errors: torch.Tensor) -> None:
        """
        Update the priorities on the sum tree
        """
        clipped_errors = np.minimum(abs_errors + self.PER_e, self.max_absolute_error)
        priority = np.power(clipped_errors, self.PER_a)
        for idx, p in zip(tree_idx, priority):
            self.SumTree.update(idx, p)

    def min_train_size_reached(self) -> bool:
        return self.ReplayBuffer.min_train_size_reached()

    def update_n_step(self, new_n_step: int, new_gamma: float = None) -> None:
        self.ReplayBuffer.update_n_step(new_n_step, new_gamma)
