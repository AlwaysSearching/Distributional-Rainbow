from typing import Literal, Union, Optional
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Define the Parameters that define the Agent training process

    Parameters
    ----------
    agent_name:
        Specify the agent to used.
    buffer_type:
        Specify the memory buffer to use.
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
    epsilon:
        Epsilon for epsilon greedy policy
    tau:
        Temperature parameter for softmax action selection
    memory_min_train_size:
        Number of transitions to observe before first policy train step.
    memory_maxlength:
        Maximum number of experiences to hold in the memory buffer.
    train_freq:
        Frequency of policy updates. Train Policy every train_freq environment steps.
    noisy_networks:
        Whether to use Noisy Layers in the DDQN.
    n_step_return:
        Number of steps to use in computing N-step returns.
    PER_a:
        PER alpha term to trade off between uniform and prioritized sampling.
    PER_b:
        PER beta value to reward more frequently sampled experiences to adjust for sampling bias.
    PER_e:
        PER epsilon value to avoid zero priority experiences.
    PER_b_increment:
        PER beta increment per step. This is used to increase the importance sampling
        weights over time to increasilngly reduce bias.
    """
    # Agent and Memory Buffer parameters
    agent_name: Union[Literal["DQN"], Literal["DDQN"], Literal["Categorical_DDQN"]]
    buffer_type: Union[Literal["PER"], Literal["Replay"]]

    # MDP Parameters
    frame_hist: int
    state_dim: int
    action_dim: int
    gamma: float
    batchsize: int

    # action sampling parameters
    tau: float
    epsilon: float

    # DQN Model/Training Parameters
    lr: float
    iterations_per_epoch: int
    train_freq: int
    noisy_networks: bool
    n_step_return: int
    clip_grad_val: float

    # Memory Buffer Parameters 
    memory_min_train_size: int
    memory_maxlength: int

    # Prioritized Experience Replay Parameters
    PER_a: Optional[float]
    PER_b: Optional[float]
    PER_e: Optional[float]
    PER_b_increment: Optional[float]
    PER_max_priority: Optional[float]
    