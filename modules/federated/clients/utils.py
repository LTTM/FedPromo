"""
This module contains utility functions for the federated learning process.

Classes:
    None

Functions:
    get_weights: Function to get the weights of a model.
    set_weights: Function to set the weights of a model.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

from typing import List, OrderedDict as OrderedDictType
from collections import OrderedDict
from copy import deepcopy as copy

import torch
import numpy as np


# Define the element to export
__all__ = [
    "get_weights",
    "set_weights",
]


def get_weights(net: OrderedDictType[str, torch.nn.Module]) -> List[np.ndarray]:
    """
    Function to get the weights of a model.

    Args:
        net (OrderedDict[str, torch.nn.Module]): The model to get the weights from.
            The key is the name of the model and the value is the model itself.

    Returns:
        Dict[str, List[np.ndarray]]: The weights of the model.
            The key is the name of the model and the value is the list of weights
    """

    params = []
    for model in net.values():
        params += [param.cpu().numpy() for param in model.state_dict().values()]
    return params


def set_weights(
    net: OrderedDictType[str, torch.nn.Module], parameters: List[np.ndarray]
) -> None:
    """
    Function to set the weights of a model.

    Args:
        net (OrderedDict[str, torch.nn.Module]): The model to set the weights to.
            The key is the name of the model and the value is the model itself.
        parameters (List[np.ndarray]): The weights to set to the model.

    Returns:
        None
    """

    # Copy the parameters to avoid modifying the original list
    parameters = copy(parameters)

    # Load the parameters in the correspondet state_dict
    for model_key, model in net.items():
        param_dict = OrderedDict()
        for param_key in model.state_dict():
            param = parameters.pop(0)
            param_dict[param_key] = torch.tensor(param)
        net[model_key].load_state_dict(param_dict, strict=True)
