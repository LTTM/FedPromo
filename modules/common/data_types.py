"""
Module containing the custom typing definitions used in this work.

Classes:
    None

Functions:
    None

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

from typing import Union


import torch
from omegaconf import DictConfig

# Define the default import behavior for the module.
__all__ = ["TorchModule", "TorchModuleObj", "TorchOptim", "TorchOptimObj"]


# Define custom typing
TorchModule = torch.nn.Module
TorchModuleObj = Union[DictConfig, torch.nn.Module]
TorchOptim = torch.optim.Optimizer
TorchOptimObj = Union[DictConfig, torch.optim.Optimizer]
