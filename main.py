import logging
from typing import Optional
from uuid import uuid4

import gymnasium as gym

from agent.base_agent import BaseAgent
from utils import generate_config


LOG = logging.getLogger(__name__)



def validation(
    env: gym.Env,
    policy: BaseAgent,
    num_episodes: int,
    render: bool = False,
    log_dir: Optional[str] = None,
):
    """Collect N episodes of experience from the trained policy.

    While collecting each rollout, we also log the value head outputs and the pixel
    state of the game for later visualization and analysis.

    Parameters
    ----------
    env : gym.Env
        OpenAI Gym environment
    policy : BaseAgent
        The trained policy used to collect the experience
    num_episodes : int
        Number of episodes to collect
    render : bool
        Whether to render the environment
    log_dir : str
        Directory to save the results to
    """
    pass  # TODO


def train(
    env: gym.Env,
    agent: BaseAgent,
    num_episodes: int,
    log_dir: Optional[str] = None,
):
    """Train the agent in the given environment.

    Parameters
    ----------
    env : gym.Env
        OpenAI Gym environment
    agent : BaseAgent
        The agent to train
    num_episodes : int
        Number of episodes to train for
    log_dir : str
        Directory to save the results to
    """
    pass  # TODO


def main():
    device, config = generate_config(LOG=LOG)
    env = gym.make(config.env_name)

    for _ in range(1000):
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
