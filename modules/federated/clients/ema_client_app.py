"""
This module contains all the elements needed to create a Flower Client instance that uses EMA.

Classes:
    EMAClient: Exponential Moving Average client implementation for PyTorch.

Functions:
    generate_client_fn: Generate the client_fn function to use for the federated learning process.

Constants:
    None

Exceptions:
    ValueError: If the param2train is not allowed.
    AttributeError: If the model does not have a 'linear' attribute.


Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union
from collections import OrderedDict
from copy import deepcopy as copy

import torch
from omegaconf import DictConfig

from modules.common.training_utils import set_optimizer
from modules.federated.clients.base_client_app import BaseClient

from .task import train
from .utils import get_weights, set_weights


# Define wahat to export
__all__ = ["EMAClient"]


# Define Flower Client and client_fn
class EMAClient(BaseClient):
    """
    EMA client implementation for PyTorch.
    This specific client implemnted EMA in the training process.

    Args:
        partition_id (int): The partition ID.
        model (torch.nn.Module): The model to train.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        trainloader (torch.utils.data.DataLoader): The training data loader.
        valloader (torch.utils.data.DataLoader): The validation data loader
    """

    def __init__(
        self,
        partition_id: int,
        model: torch.nn.Module,
        head: torch.nn.Module,
        device: Union[torch.device, str],
        criterion: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, DictConfig],
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        **kwargs,
    ):

        # Call the parent constructor
        super().__init__(
            partition_id=partition_id,
            model=model,
            head=head,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
        )

        # Define the static model variables
        self.ema_model = copy(model)
        self.ema_model = self.ema_model.to(device)
        self.ema_head = copy(head)
        self.ema_head = self.ema_head.to(device)

    def fit(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Set the model parameters
        models = OrderedDict({"model": self.model, "head": self.head})
        set_weights(models, parameters)

        # Set the ema model parameters
        ema_models = OrderedDict({"model": self.ema_model, "head": self.ema_head})
        set_weights(ema_models, parameters)

        # Define which model will be trained
        # First check if the model2train is one of the allowed ones
        optimizer, module2train = set_optimizer(
            model=self.model,
            head=self.head,
            optimizer=self.optimizer,
            module2train=config["module2train"],
            lr=config["lr"],
        )

        # Define the models2train dict
        models2train = OrderedDict(
            {
                "model": self.model,
                "head": self.head,
            }
        )

        # Train the model
        metrics = train(
            models=models2train,
            module2train=module2train,
            device=self.device,
            criterion=self.criterion,
            optimizer=optimizer,
            dataloader=self.trainloader,
            epochs=config["local_epochs"],
            train_config={
                "type": "base",
                "class_debias": config["class_debias"],
                "class_debias_weight": config["class_debias_weight"]
            },
        )

        # Parse ema decay
        if "ema_decay" not in config or config["ema_decay"] is None:
            config["ema_decay"] = 0.5

        # Perform EMA update
        updated_params = OrderedDict()
        for (param_name, param), ema_param in zip(
            self.model.state_dict().items(), self.ema_model.state_dict().values()
        ):
            updated_params[param_name] = (
                ema_param * (1 - config["ema_decay"]) + param * config["ema_decay"]
            )
        self.model.load_state_dict(updated_params)

        updated_params = OrderedDict()
        for (param_name, param), ema_param in zip(
            self.head.state_dict().items(), self.ema_head.state_dict().values()
        ):
            updated_params[param_name] = (
                ema_param * (1 - config["ema_decay"]) + param * config["ema_decay"]
            )
        self.head.load_state_dict(updated_params)

        models = OrderedDict({"model": self.model, "head": self.head})
        return get_weights(models), len(self.trainloader), metrics
