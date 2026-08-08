"""
This module contains all the elements needed to create a Flower Client instance that uses
the Moon algorithm in the training process.

Classes:
    MoonClient: Moon client implementation for PyTorch.

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
from pathlib import Path

import torch
from omegaconf import DictConfig

from modules.common.training_utils import set_optimizer
from modules.federated.clients.base_client_app import BaseClient

from modules.federated.clients.task import train
from modules.federated.clients.utils import get_weights, set_weights


# Define wahat to export
__all__ = ["MoonClient"]


# Define Flower Client and client_fn
class MoonClient(BaseClient):
    """
    Moon client implementation for PyTorch.
    This specific client implemnted Moon in the training process.

    Args:
        partition_id (int): The partition ID.
        model (torch.nn.Module): The model to train.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        trainloader (torch.utils.data.DataLoader): The training data loader.
        valloader (torch.utils.data.DataLoader): The validation data loader.
        save_path (Union[str, Path]): The path where to save the model.
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
        save_path: Union[str, Path],
        **kwargs, # pylint: disable=unused-argument
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

        # Parse the save path
        if isinstance(save_path, str):
            self.save_path = Path(save_path)
        else:
            self.save_path = save_path
        self.save_path = self.save_path / f"prev_clients_checkpoint/{self.partition_id}"
        self.save_path.mkdir(parents=True, exist_ok=True)

    def fit(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Set the model parameters
        models = OrderedDict({"model": self.model, "head": self.head})
        set_weights(models, parameters)

        # Define all the other models
        prev_model = copy(self.model)
        prev_head = copy(self.head)
        global_model = copy(self.model)
        global_head = copy(self.head)

        if (self.save_path / "model_prev_state.pt").exists():
            prev_model.load_state_dict(
                torch.load(self.save_path / "model_prev_state.pt", weights_only=False)
            )
        if (self.save_path / "head_prev_state.pt").exists():
            prev_head.load_state_dict(
                torch.load(self.save_path / "head_prev_state.pt", weights_only=False)
            )

        # Define the models2train dict
        models2train = OrderedDict(
            {
                "model": self.model,
                "prev_model": prev_model,
                "head": self.head,
                "prev_head": prev_head,
                "global_model": global_model,
                "global_head": global_head,
            }
        )

        # Define which model will be trained
        # First check if the model2train is one of the allowed ones
        optimizer, module2train = set_optimizer(
            model=self.model,
            head=self.head,
            optimizer=self.optimizer,
            module2train=config["module2train"],
            lr=config["cur_lr"],
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
                "type": "moon",
                "mu": config["mu"],
                "temperature": config["temperature"],
                "class_debias": config["class_debias"],
                "class_debias_weight": config["class_debias_weight"]
            },
        )

        # Append to the metrics the current lr
        metrics["lr"] = config["cur_lr"]

        # Save the new prev model
        torch.save(
            self.model.state_dict(),
            self.save_path / "model_prev_state.pt",
        )
        torch.save(
            self.head.state_dict(),
            self.save_path / "head_prev_state.pt",
        )

        models = OrderedDict({"model": self.model, "head": self.head})
        return get_weights(models), len(self.trainloader), metrics
