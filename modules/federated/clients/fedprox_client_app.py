"""
This module contains all the elements needed to create a Flower Client instance that uses
FedProx in training.

Classes:
    FedProx: FedProx client implementation for PyTorch.

Functions:
    None

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

from modules.federated.clients.task import train
from modules.federated.clients.utils import get_weights, set_weights


# Define wahat to export
__all__ = ["FedProxClient"]


# Define Flower Client and client_fn
class FedProxClient(BaseClient):
    """
    FedProx client implementation for PyTorch.
    This specific client implemnted a FedProx strategy.

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
        self.last_round_model = copy(model)
        self.last_round_model = self.last_round_model.to(device)
        self.last_round_head = copy(head)
        self.last_round_head = self.last_round_head.to(device)

    def fit(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Set the model parameters
        models = OrderedDict({"model": self.model, "head": self.head})
        set_weights(models, parameters)

        # Set the same weight to the last_round_model
        self.last_round_model.load_state_dict(self.model.state_dict())
        self.last_round_head.load_state_dict(self.head.state_dict())

        # Define which model will be trained
        # First check if the model2train is one of the allowed ones
        optimizer, module2train = set_optimizer(
            model=self.model,
            head=self.head,
            optimizer=self.optimizer,
            module2train=config["module2train"],
            lr=config["cur_lr"],
        )

        # Define the models2train dict
        models2train = OrderedDict(
            {
                "model": self.model,
                "last_round_model": self.last_round_model,
                "head": self.head,
                "last_round_head": self.last_round_head,
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
                "type": "fedprox",
                "proximal_mu": config["proximal_mu"],
                "class_debias": config["class_debias"],
                "class_debias_weight": config["class_debias_weight"]
            },
        )

        # Append to the metrics the current lr
        metrics["lr"] = config["cur_lr"]

        models = OrderedDict({"model": self.model, "head": self.head})
        return get_weights(models), len(self.trainloader), metrics
