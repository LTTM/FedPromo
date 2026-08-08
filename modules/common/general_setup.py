"""
Module to set up the general configuration for the computations.

Classes:
    None

Functions:
    set_device: Set the device to use for the computations.
    set_seed: Set the seed for the random number generators.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

import random

import torch
import numpy as np

from .logger import Logger


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


__all__ = ["set_device", "set_seed"]


def set_device() -> torch.device:
    """
    Set the device to use for the computations.

    Args:
        None
    Returns:
        torch.device: The device to use for the computations.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Log the device to use
    logger.info("Device set to: %s", device)

    return device


def set_seed(seed: int) -> None:
    """
    Set the seed for the random number generators.

    Args:
        seed (int): The seed to use.
    Returns:
        torch.generator: The generator to use.
    """

    # Set the seed for the random number generators
    # Torch
    torch_gen = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    logger.info("Seed set to: %d", seed)

    return torch_gen
