"""
Module containing utility functions that could be usefull for all the trainer modules.

Classes:
    None

Functions:
    set_optimizer: Check if the module2train is allowed and load the corresponding optimizer.

Constants:
    None

Exceptions:
    ValueError: If the module2train is not allowed.
    ValueError: If the mode is not 'train' or 'eval'.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Optional, Union, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from modules.common.constants import ALLOWED_MODULE2TRAIN
from modules.common.data_types import TorchModule, TorchOptimObj


# Define the default import behavior for the module.
__all__ = ["set_optimizer", "set_train_mode", "set_eval_mode", "normalize_tensor"]


def set_optimizer(
    model: TorchModule,
    head: Union[TorchModule, Dict[str, TorchModule]],
    optimizer: TorchOptimObj,
    module2train: Optional[str] = None,
    lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """
    Check if the module2train is allowed.
    Then load the corresponding optimizer.

    Args:
        model (TorchModule): The model to train.
        head (Union[TorchModule, Dict[str, TorchModule]]): The head of the model.
            Or a dict containing the different heads of the model.
        optimizer (TorchOptimObj): The optimizer to use for the training.
            (Not initialized)
        module2train (Optional[str]): The module to train.
            Default is None -> load everything.
        lr (Optional[float]): The learning rate to use for the optimizer
            It will override the learning rate in the optimizer.
            Default is None.

    Raises:
        ValueError: If the module2train is not allowed.

    Returns:
        Tuple[torch.optim.Optimizer, str]: The optimizer and the string containing
            the module to train.
    """

    if isinstance(head, dict):
        head_params = []
        for h in head.values():
            head_params += list(h.parameters())
    else:
        head_params = head.parameters()

    # Set the correct model parameters in the optimizer
    if module2train == "all" or module2train is None:
        # Define a param_group for model and head
        params = list(model.parameters()) + list(head_params)
    elif module2train == "main":
        # Define a param_group for just the model
        params = model.parameters()
    elif module2train == "head":
        # Define a param_group for just the head
        params = head_params
    else:
        raise ValueError(f"module2train must be one of {ALLOWED_MODULE2TRAIN}")

    # Define the optimizer
    if isinstance(optimizer, DictConfig) and lr is not None:
        optimizer = instantiate(optimizer, lr=lr, params=params)
    elif isinstance(optimizer, DictConfig):
        optimizer = instantiate(optimizer, params=params)
    else:
        optimizer = optimizer(lr=lr, params=params)

    return optimizer, module2train


def set_train_mode(
    model: TorchModule,
    head: Union[TorchModule, Dict[str, TorchModule]],
    module2train: str,
    s_model: Optional[TorchModule] = None,
) -> None:
    """
    Set the modules in training or eval mode according to the module2train

    Args:
        model (TorchModule): The client model.
        head (Union[TorchModule, Dict[str, TorchModule]]): The head of the model.
            Or a dict containing the different heads of the model.
        module2train (str): The module to train.
        s_model (Optional[TorchModule]): The server model.
            Always put in eval mode.
            Default is None.

    Returns:
        None
    """

    if s_model is not None:
        s_model.eval()

    if module2train == "all":
        model.train()
        _set_head_mode(head, "train")
    elif module2train == "main":
        model.train()
        _set_head_mode(head, "eval")
    elif module2train == "head":
        model.eval()
        _set_head_mode(head, "train")
    else:
        raise ValueError(f"module2train must be one of {ALLOWED_MODULE2TRAIN}")


def set_eval_mode(
    model: TorchModule,
    head: Union[TorchModule, Dict[str, TorchModule]],
    s_model: Optional[TorchModule] = None,
) -> None:
    """
    Set all the modules in eval mode.

    Args:
        model (TorchModule): The client model.
        head (Union[TorchModule, Dict[str, TorchModule]]): The head of the model.
            Or a dict containing the different heads of the model.
        s_model (Optional[TorchModule]): The server model.
            Default is None.

    Returns:
        None
    """

    model.eval()
    _set_head_mode(head, "eval")
    if s_model is not None:
        s_model.eval()


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to have norm 1.
    Args:
        tensor (Tensor): The tensor to normalize.
        Returns:
            Tensor: The normalized tensor.
    """
    norm = torch.norm(tensor.detach(), dim=-1, keepdim=True) + 1e-8
    return tensor / norm


def _set_head_mode(
    head: Union[TorchModule, Dict[str, TorchModule]],
    mode: str,
) -> None:
    """
    Set the head or heads to the given mode.

    Args:
        head (Union[TorchModule, Dict[str, TorchModule]]): The head of the model.
            Or a dict containing the different heads of the model.
        mode (str): The mode to set the head to.
            train or eval.

    Returns:
        None
    """

    # Check if the mode is correct
    if not mode in ["train", "eval"]:
        raise ValueError("mode must be one of ['train', 'eval']")

    # Set the head or heads to the given mode
    if isinstance(head, dict):
        for h in head.values():
            getattr(h, mode)()
    else:
        getattr(head, mode)()
