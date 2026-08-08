"""
Common functions for flower local differential privacy (LDP) implementation.

Author: Matteo Caligiuri
"""

import numpy as np
from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.differential_privacy import get_norm, compute_stdv


# Overrides of specific local DP utils functions
def add_localdp_gaussian_noise_to_params(
    model_params: Parameters,
    sensitivity: float,
    epsilon: float,
    delta: float,
    module2train: str,
    n_head_layers: int,
) -> Parameters:
    """
    Add local DP gaussian noise to model parameters.

    Args:
        model_params (Parameters): The model parameters to add noise to.
        sensitivity (float): The sensitivity of the model.
        epsilon (float): The privacy budget.
        delta (float): The privacy budget.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        Parameters: The model parameters with added noise.
    """
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon,
        module2train,
        n_head_layers,
    )
    return ndarrays_to_parameters(model_params_ndarrays)


def add_gaussian_noise_inplace(
    input_arrays: NDArrays, std_dev: float, module2train: str, n_head_layers: int
) -> None:
    """
    Add Gaussian noise to each element of the input arrays.

    Args:
        input_arrays (NDArrays): The input arrays to add noise to.
        std_dev (float): The standard deviation of the Gaussian noise.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        None: The input arrays are modified in place.
    """

    for index, array in enumerate(input_arrays):
        if module2train == "all" and array.dtype != int:
            # Add gaussian noise to all the model only
            # if context.run_config["module2train"] is set to all
            # Add the Gaussian noise only if the dtype is not int
            array += np.random.normal(0, std_dev, array.shape)
        elif (
            module2train == "main"
            and index < len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the main module
            # every array except the last n_head_layers
            # Add the Gaussian noise only if the dtype is not int
            array += np.random.normal(0, std_dev, array.shape)
        elif (
            module2train == "head"
            and index >= len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the head module
            # the last n_head_layers
            # Add the Gaussian noise only if the dtype is not int
            array += np.random.normal(0, std_dev, array.shape)
        else:
            # Do not add noise to the model params
            pass


def add_gaussian_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
    module2train: str,
    n_head_layers: int,
) -> Parameters:
    """
    Add gaussian noise to model parameters.

    Args:
        model_params (Parameters): The model parameters to add noise to.
        noise_multiplier (float): The noise multiplier.
        clipping_norm (float): The clipping norm.
        num_sampled_clients (int): The number of sampled clients.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        Parameters: The model parameters with added noise.
    """

    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
        module2train,
        n_head_layers,
    )
    return ndarrays_to_parameters(model_params_ndarrays)


def compute_clip_model_update(
    param1: NDArrays,
    param2: NDArrays,
    clipping_norm: float,
    module2train: str,
    n_head_layers: int,
) -> None:
    """
    Compute model update (param1 - param2) and clip it.

    Then add the clipped value to param1.

    Args:
        param1 (NDArrays): The first set of parameters.
        param2 (NDArrays): The second set of parameters.
        clipping_norm (float): The clipping norm.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        None: The first set of parameters is modified in place.
    """

    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    clip_inputs_inplace(model_update, clipping_norm, module2train, n_head_layers)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]


def clip_inputs_inplace(
    input_arrays: NDArrays, clipping_norm: float, module2train: str, n_head_layers: int
) -> None:
    """
    Clip model update based on the clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/abs/1710.06963

    Args:
        input_arrays (NDArrays): The input arrays to clip.
        clipping_norm (float): The clipping norm.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        None: The input arrays are modified in place.
    """

    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for index, array in enumerate(input_arrays):
        if module2train == "all" and array.dtype != int:
            # Add gaussian noise to all the model only
            # if context.run_config["module2train"] is set to all
            array *= scaling_factor
        elif (
            module2train == "main"
            and index < len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the main module
            # every array except the last n_head_layers
            array *= scaling_factor
        elif (
            module2train == "head"
            and index >= len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the head module
            # the last n_head_layers
            array *= scaling_factor
        else:
            # Do not add noise to the model params
            pass


def adaptive_clip_inputs_inplace(
    input_arrays: NDArrays, clipping_norm: float, module2train: str, n_head_layers: int
) -> bool:
    """
    Clip model update based on the clipping norm in-place.

    It returns true if scaling_factor < 1 which is used for norm_bit
    FlatClip method of the paper: https://arxiv.org/abs/1710.06963

    Args:
        input_arrays (NDArrays): The input arrays to clip.
        clipping_norm (float): The clipping norm.
        module2train (str): The module to train. Can be "all", "main" or "head".
        n_head_layers (int): The number of head layers in the model.
            Used only if module2train is set to "main" or "head".

    Returns:
        bool: True if scaling_factor < 1, False otherwise.
    """

    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for index, array in enumerate(input_arrays):
        if module2train == "all" and array.dtype != int:
            # Add gaussian noise to all the model only
            # if context.run_config["module2train"] is set to all
            array *= scaling_factor
        elif (
            module2train == "main"
            and index < len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the main module
            # every array except the last n_head_layers
            array *= scaling_factor
        elif (
            module2train == "head"
            and index >= len(input_arrays) - n_head_layers
            and array.dtype != int
        ):
            # Add Gaussian noise only to the head module
            # the last n_head_layers
            array *= scaling_factor
        else:
            # Do not add noise to the model params
            pass
    return scaling_factor < 1
