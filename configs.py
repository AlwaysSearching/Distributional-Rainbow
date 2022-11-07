from typing import Literal, Union, Optional
from pydantic import BaseModel
from torch import nn


class AgentConfig(BaseModel):
    """Define the Parameters that define the Agent training process

    Parameters
    ----------
    agent:
        Specify the agent to used.
    buffer:
        specify the memory buffer to use.
    frame_hist:
        Number of frames to stack together to form a state.
    state_dim:
        Dimension of the state/observation space
    action_dim:
        Dimension of the action space. For DQN & DDQN this must be an integer value.
    gamma:
        Discount Factor of the chosen MDP
    batchsize:
        Training batch Size
    lr:
        Learning Rate for the Q-Networks
    iterations_per_epoch:
        Number of training iterations per epoch
    clip_grad_val:
        Gradient clipping value
    train_freq:
        Frequency of policy updates. Train Policy every train_freq environment steps.
    noisy_networks:
        Whether to use Noisy Layers in the DDQN.
    update_ratio:
        The ratio to use in the polyak averaging of the target network.
    n_step_return:
        Number of steps to use in computing N-step returns.
    epsilon:
        Epsilon for epsilon greedy policy
    memory_min_train_size:
        Number of transitions to observe before first policy train step.
    memory_maxlength:
        Maximum number of experiences to hold in the memory buffer.
    PER_a:
        PER alpha term to trade off between uniform and prioritized sampling.
    PER_b:
        PER beta value to reduce the weight of more frequently sampled experiences
        to adjust for sampling bias.
    PER_e:
        PER epsilon value to avoid zero priority experiences.
    PER_b_increment:
        PER beta increment per step. This is used to increase the importance sampling
        weights over time to increasilngly reduce bias.
    PER_max_priority:
        Maximum priority value to use in PER.
    num_atoms:
        Number of atoms to use in the Categorical DQN
    v_min:
        Minimum value of the support
    v_max:
        Maximum value of the support
    """
    # Agent and Memory Buffer parameters
    agent_name: Union[Literal["DQN"], Literal["DDQN"], Literal["Categorical_DDQN"]]
    buffer_type: Union[Literal["PER"], Literal["Replay"]]
    network: nn.Module

    # MDP Parameters
    frame_hist: int
    state_dim: int
    action_dim: int
    gamma: float

    # DQN Model/Training Parameters
    batchsize: int
    lr: float
    iterations_per_epoch: int
    clip_grad_val: Optional[float]
    train_freq: int
    noisy_networks: bool
    n_step_return: int

    # action sampling parameters
    epsilon: float

    # Memory Buffer Parameters
    memory_min_train_size: int
    memory_maxlength: int

    # Prioritized Experience Replay Parameters
    PER_a: Optional[float]
    PER_b: Optional[float]
    PER_e: Optional[float]
    PER_b_increment: Optional[float]
    PER_max_priority: Optional[float]

    # Categorical DDQN Parameters
    num_atoms: Optional[int]
    v_min: Optional[float]
    v_max: Optional[float]
