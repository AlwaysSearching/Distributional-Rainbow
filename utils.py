import logging
import argparse
import os

import numpy as np
import torch

from configs import AgentConfig


def generate_config(LOG: logging.logger):
    """Generate the config for the experiment from the command line args.

    Parameters
    ----------
    LOG : logging.logger
        Logger to log the experiment details
    TEST_ID : str
        Unique ID for the experiment
    """

    parser = argparse.ArgumentParser(
        description="Distributional RL Agent Training and Evaluation"
    )

    parser.add_argument(
        "--seed", type=int, default=int, help="Random seed for the experiment"
    )
    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument(
        "--environment", type=str, default="LunarLander-v2", help="Environment to use"
    )
    parser.add_argument(
        "--agent", type=str, default="Categorical_DDQN", help="Agent to use"
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default="PrioritizedExperienceReplay",
        help="Memory Buffer to use",
    )
    parser.add_argument(
        "--frame_hist",
        type=int,
        default=4,
        help="number of frames to stack to form a state",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for the environment"
    )
    parser.add_argument(
        "--batchsize", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the Q-Networks"
    )
    parser.add_argument(
        "--iteration_per_epoch",
        type=int,
        default=8,
        help="Number of batches to train over for each training step",
    )
    parser.add_argument(
        "--clip_grad_value", type=float, default=10.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--train_freq",
        type=int,
        default=16,
        help="Number of environment steps to take before training the policy",
    )
    parser.add_argument(
        "--noisy_networks",
        type=bool,
        default=True,
        help="Whether to use Noisy Layers in the DDQN",
    )
    parser.add_argument(
        "--update_ratio",
        type=float,
        default=0.2,
        help="The ratio to use in the polyak averaging of the target network",
    )
    parser.add_argument(
        "--n_step_returns",
        type=int,
        default=5,
        help="Number of steps to use in computing N-step returns",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Initial value of epsilon for the epsilon-greedy policy",
    )
    parser.add_argument(
        "--memory_min_train_size",
        type=int,
        default=516,
        help="Minimum number of samples in the replay buffer before training",
    )
    parser.add_argument(
        "--memory_max_length",
        type=int,
        default=5e4,
        help="Maximum number of samples in the replay buffer",
    )
    parser.add_argument(
        "--PER_a", type=float, default=0.5, help="Priority experience exponent"
    )
    parser.add_argument(
        "--PER_b",
        type=float,
        default=0.4,
        help="Priority experience importance sampling weight",
    )
    parser.add_argument(
        "--PER_e",
        type=float,
        default=5e-2,
        help="Priority experience minimum sample priority",
    )
    parser.add_argument(
        "--PER_b_increment",
        type=float,
        default=2.5e-6,
        help="Priority experience beta increment size per training step",
    )
    parser.add_argument(
        "--PER_max_priority",
        type=float,
        default=5,
        help="Priority experience maximum sample priority",
    )
    parser.add_argument(
        "--num_atoms",
        type=int,
        default=51,
        help="Number of atoms in the discritized value distribution",
    )
    parser.add_argument(
        "--v_min",
        type=float,
        default=-10,
        help="Minimum value in the discritized value distribution",
    )
    parser.add_argument(
        "--v_max",
        type=float,
        default=10,
        help="Maximum value in the discritized value distribution",
    )
    parser.add_argument(
        "--v_max",
        type=float,
        default=10,
        help="Maximum value in the discritized value distribution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Maximum value in the discritized value distribution",
    )

    args = parser.parse_args()

    results_dir = os.path.join("results", args.id)
    if not os.path.exists(results_dir):
        LOG.info(f"Creating directory {results_dir}")
        os.makedirs(results_dir)
    LOG.info(f"Logging results to {results_dir}")

    np.random.seed(args.seed)
    torch_seed = np.random.randint(1, 10000)
    torch.manual_seed(torch_seed)
    LOG.info(
        f"""
        Setting numpy random seed to {args.seed}
        Setting torch random seed to {torch_seed}
        """
    )

    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
    else:
        device = torch.device("cpu")
    LOG.info(f"Torch device set to {device}")

    return device, AgentConfig(**args.to_dict())
