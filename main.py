import logging
import argparse
from typing import Optional
from uuid import uuid4

import gymnasium as gym

from agent.base_agent import BaseAgent
from configs import AgentConfig


LOG = logging.getLogger(__name__)
TEST_ID = str(uuid4()).replace("-", "_")

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()


def generate_config():
    """Generate the config for the experiment from the command line args."""

    parser = argparse.ArgumentParser(description="Distributional RL Agent Training and Evaluation")

    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 
    parser.add_argument('--id', type=str, default='default', help='Experiment ID') 


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
    pass # TODO  


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
    pass # TODO



def main():
    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
