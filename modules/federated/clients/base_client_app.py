"""
This module contains all the elements needed to create a basic Flower Client instance

Classes:
    FlowerClient: Flower client implementation for PyTorch.

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

from typing import Union, Type
from collections import OrderedDict

import torch
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig

from modules.common.training_utils import set_optimizer

from .task import train, test
from .utils import get_weights, set_weights


# Define wahat to export
__all__ = ["generate_client_fn", "BaseClient"]


# Define Flower Client and client_fn
class BaseClient(NumPyClient):
    """
    Basic flower client implementation for PyTorch.

    Args:
        partition_id (int): The partition ID.
        model (torch.nn.Module): The model to train.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        trainloader (torch.utils.data.DataLoader): The training data loader.
        valloader (torch.utils.data.DataLoader): The validation data loader.
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
        **kwargs,  # pylint: disable=unused-argument
    ):

        # Initialize the FlowerClient variables
        self.partition_id = partition_id
        self.model = model
        self.head = head
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        models = OrderedDict({"model": self.model, "head": self.head})
        return get_weights(models)

    def fit(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Set the model parameters
        models = OrderedDict({"model": self.model, "head": self.head})
        set_weights(models, parameters)

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
        models2train = {"model": self.model, "head": self.head}

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

        # Append to the metrics the current lr
        metrics["lr"] = config["cur_lr"]

        models = OrderedDict({"model": self.model, "head": self.head})
        return get_weights(models), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] evaluate, config: {config}")

        # Set the model parameters
        models = OrderedDict({"model": self.model, "head": self.head})
        set_weights(models, parameters)

        # Define the models2train dict
        models2train = {"model": self.model, "head": self.head}

        # Evaluate the model
        metrics = test(
            models=models2train,
            device=self.device,
            criterion=self.criterion,
            dataloader=self.valloader,
        )

        # Extract the loss from the metrics
        loss = metrics.pop("loss")
        return float(loss), len(self.valloader), metrics


def generate_client_fn(
    client_type: Type[NumPyClient],
    model: torch.nn.Module,
    head: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, DictConfig],
    dataloaders: callable,
    save_path: str,
) -> callable:
    """
    Generate the client_fn function to use for the federated learning process.

    Args:
        clinet_type (Type[NumPyClient]): The type of client to use.
        model (torch.nn.Module): The model to train.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        dataloaders (callable): The function used to load the data
        save_path (str): The path where to save data of the current run.

    Returns:
        callable: The client_fn function to use for the federated learning process.
    """

    def client_fn(context: Context) -> NumPyClient:
        """
        Client function used to create a Flower Client instance.

        Args:
            context (flwr.common.Context): The context used to create the Flower Client instance.

        Returns:
            NumPyClient: The Flower Client instance.
        """

        # Load model and data
        partition_id = context.node_config["partition-id"]
        data = dataloaders(partition_id)

        # Return Client instance
        return client_type(
            partition_id=partition_id,
            model=model,
            head=head,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            trainloader=data["train"],
            valloader=data["val"],
            save_path=save_path,
        ).to_client()

    return client_fn
