"""
Module for utility functions used by the federated strategies.

Classes:
    None

Functions:
    set_strategy: Set the strategy to use for the federated learning process.
    get_on_fit_config: Get the function to configure the server's training process.
    get_fit_metrics_aggregation_fn: Get the function to aggregate the training metrics.
    get_evaluate_fn: Get the function to evaluate the model on the test set.
    get_on_evaluate_config: Get the function to configure the server's evaluation process.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union, Optional, Callable, List, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
from importlib import import_module
from logging import INFO
from contextlib import nullcontext
from copy import deepcopy as copy
import random

import numpy as np
import torch
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    NDArrays,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.differential_privacy import get_norm
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
import mlflow
from matplotlib import pyplot as plt

from modules.federated.clients.utils import set_weights
from modules.common.training_utils import (
    set_train_mode,
    set_eval_mode,
    normalize_tensor,
)
from modules.common.logger import log2logger


# Define the element to export
__all__ = [
    "set_strategy",
    "get_on_fit_config",
    "get_fit_metrics_aggregation_fn",
    "get_evaluate_fn",
    "get_on_evaluate_config",
]


def set_strategy(
    strategy: str,
    p_client_dropout: float = 0.0,
    model: Optional[torch.nn.Module] = None,
    head: Optional[torch.nn.Module] = None,
    dts_name: Optional[str] = None,
    save_every: Optional[int] = 1,
    save_path: Optional[Union[str, Path]] = None,
    data_logger: Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]]] = None,
    num_tot_rounds: Optional[int] = None,
    class_debias: bool = False,
    class_debias_weight: float = 1.0,
) -> Callable:
    """
    Set the strategy to use for the federated learning process, ensuring that the
    aggregate_fit method is customized to save the model after aggregation.

    Args:
        strategy (str): Full import path to the strategy class
            (e.g., "flwr.server.strategy.FedAvg").
        p_client_dropout (float): Probability of client dropout.
            Defaults to 0.0, meaning no dropout.
        model (Optional[torch.nn.Module], optional): The client model to save. Defaults to None.
        head (Optional[torch.nn.Module], optional): The head of the model. Defaults to None.
        dts_name (Optional[str], optional): Name of the dataset. Defaults to None.
            Used to name the saved model.
        save_every (int, optional): Number of rounds before saving the model. Defaults to 1.
        save_path (Union[str, Path], optional): Path to save the model. Defaults to None.
        data_logger (Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]], optional):
            The data logger. Defaults to None.
        num_tot_rounds (Optional[int], optional): The total number of rounds. Defaults to None.
        class_debias (bool, optional): Whether to remove the bias component from the head weights.
        class_debias_weight (float, optional): Weight to be applied to (server) debiasing step.

    Returns:
        Callable: The customized strategy class.
    """

    # Load the strategy class dynamically
    strategy_class = _load_strategy(strategy)

    # Convert save_path to Path object if it's a string
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    elif save_path is None or model is None or head is None:
        # No customization needed
        return strategy_class

    # Define the custom aggregate_fit method
    # Call the base class method and save the model after aggregation
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Log the parameters of the server before aggregation and the ones of the clients
        global_model_params = parameters_to_ndarrays(self.last_global_parameters)
        clients_prams = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        # Compute the median of the L2 norm of the difference
        # between the client weights and the global weights
        # This is done only if the server_round > 1
        # and so the last global model parameters are available
        update_norms = [
            get_norm(
                [np.subtract(x, y) for (x, y) in zip(client_prams, global_model_params)]
            )
            for client_prams in clients_prams
        ]
        median = np.median(update_norms)

        # Remove the vars used for the computation of the norms
        # to free memory space
        del clients_prams, global_model_params, update_norms

        # Call base class method
        aggregated_parameters, aggregated_metrics = strategy_class.aggregate_fit(
            self, server_round, results, failures
        )

        # Remove bias component from head weights
        if class_debias:
            # Convert the paraemters from bytes to numpy arrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Extract the last linear layer weights
            classifier_weights = aggregated_ndarrays[-1]  # Last layer weights (C x F)

            # Remove the bias component
            with torch.no_grad():
                classifier_weights -= class_debias_weight * classifier_weights.mean(
                    axis=0, keepdims=True
                )

            # Update the aggregated parameters
            aggregated_ndarrays[-1] = classifier_weights

            # Convert the numpy arrays to parameters bytes
            aggregated_parameters = fl.common.ndarrays_to_parameters(
                aggregated_ndarrays
            )

        # Log the round results
        log(
            INFO,
            "aggregate_fit (round %s): "
            + "average client loss: %.4f | average client accuracy: %3.2f%% ",
            server_round,
            aggregated_metrics["loss"],
            aggregated_metrics["accuracy"] * 100,
        )

        # Log the median of the L2 norm of the difference
        aggregated_metrics["client_dp_norm"] = median

        # Log the losses and KD loss on tensorboard or MLFlow
        if data_logger is not None:
            with (
                mlflow.start_run(
                    run_id=data_logger["mlflow"]["run_id"],
                    run_name=data_logger["mlflow"]["run_name"],
                    log_system_metrics=False,
                )
                if list(data_logger.keys())[0] == "mlflow"
                else nullcontext()
            ):
                data = {}

                data["Federated/Train Client accuracy"] = (
                    aggregated_metrics["accuracy"] * 100
                )
                data["Federated/Train Client loss"] = aggregated_metrics["loss"]

                keys_to_log = set(aggregated_metrics).difference(["accuracy", "loss"])
                for key in keys_to_log:
                    data[f"Federated/Train {key}"] = aggregated_metrics[key]

                log2logger(
                    logger=data_logger,
                    epoch=(server_round),
                    data=data,
                )

        if aggregated_parameters is not None and (
            (server_round % save_every == 0 and save_every > 0)
            or (server_round == num_tot_rounds)
        ):
            log(
                INFO, "[SERVER]: Saving round %s aggregated_parameters...", server_round
            )

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            net = {"model": model, "head": head}
            set_weights(net, aggregated_ndarrays)

            # Save the model to disk
            if dts_name is not None:
                ckpt_name = (
                    f"{server_round:03d}_{model.__class__.__name__.lower()}"
                    + f"_{dts_name.lower()}_federated_checkpoint.pth"
                )
            else:
                ckpt_name = (
                    f"{server_round:03d}_{model.__class__.__name__.lower()}"
                    + "_federated_checkpoint.pth"
                )
            ckpt_path = save_path / ckpt_name
            torch.save({k: m.state_dict() for k, m in net.items()}, ckpt_path)

            # If MLFlow is used save the model there
            if list(data_logger.keys())[0] == "mlflow" and (
                server_round == num_tot_rounds
            ):
                tmp_net = copy(net["model"])
                tmp_net.head = copy(net["head"])
                with (
                    mlflow.start_run(
                        run_id=data_logger["mlflow"]["run_id"],
                        run_name=data_logger["mlflow"]["run_name"],
                        log_system_metrics=False,
                    )
                    if list(data_logger.keys())[0] == "mlflow"
                    else nullcontext()
                ):
                    mlflow.pytorch.log_model(tmp_net, ckpt_name, signature=False)

            log(INFO, "[SERVER]: Model %s saved.", ckpt_path.name)

        return aggregated_parameters, aggregated_metrics

    # Define a custom configure_fit method
    # In this modified version it is possible to simulate
    # clients that do not send updates wit ha spcific probability
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # Call base class method
        clients = strategy_class.configure_fit(
            self, server_round, parameters, client_manager
        )

        # Drop clients with probability p_client_dropout
        if self.p_client_dropout > 0.0:
            final_clients = []
            for client in clients:
                if random.random() < 1 - self.p_client_dropout:
                    final_clients.append(client)
        else:
            final_clients = clients

        # Save the parameters as the last global model parameters
        self.last_global_parameters = copy(parameters)

        # Return client/config pairs
        return final_clients

    # Create the custom strategy class dynamically
    CustomStrategy = type(  # pylint: disable=invalid-name
        f"{strategy_class.__name__}",
        (strategy_class,),
        {
            "aggregate_fit": aggregate_fit,
            "configure_fit": configure_fit,
            "p_client_dropout": p_client_dropout,
            "last_global_parameters": None,
        },
    )

    return CustomStrategy


def _load_strategy(strategy: str) -> Strategy:
    """
    Import a strategy dynamically from its full module path.

    Args:
        strategy (str): Full path of the strategy (e.g., "flwr.server.strategy.FedAvg").

    Returns:
        object: The imported strategy class or function.
    """

    module_name = ".".join(strategy.split(".")[:-1])
    class_name = strategy.split(".")[-1]

    # Import the module dynamically
    module = import_module(module_name)

    # Get the class or function from the module
    return getattr(module, class_name)


def get_on_fit_config(config: DictConfig) -> Callable[[int], Dict]:
    """
    Get the function to configure the server's training process.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        Callable[int, Dict]: The function to configure the server's training process.
    """

    # Parse the configuration to a dict
    try:
        config = dict(config)
    except TypeError:
        config = {}

    def fit_config_fn(server_round: int) -> Dict:

        # Schedule the lr based on the round number
        if config["use_lr_scheduler"]:
            scale = (
                np.cos(np.pi * server_round / (2 * config["scheduler_tot_rounds"])) ** 2
            )
            config["cur_lr"] = float(config["lr"] * scale)

            # Log the scheduler scale
            config["scheduler_scale"] = scale
        else:
            config["cur_lr"] = config["lr"]
            config["scheduler_scale"] = None

        return {
            **config,
            "server_round": server_round,
        }

    return fit_config_fn


def get_on_evaluate_config(config: DictConfig) -> Callable[[int], Dict]:
    """
    Get the function to configure the server's evaluation process.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        Callable[int, Dict]: The function to configure the server's training process.
    """

    # Parse the configuration to a dict
    try:
        config = dict(config)
    except TypeError:
        config = {}

    def evaluate_config_fn(server_round: int) -> Dict:

        return {
            **config,
            "server_round": server_round,
        }

    return evaluate_config_fn


def get_fit_metrics_aggregation_fn() -> Callable[[List[Tuple[int, Dict]]], Dict]:
    """
    Get the function to aggregate the training metrics.

    Args:
        None

    Returns:
        Callable[[List[Tuple[int, Dict]]], Dict]: The function to aggregate the training metrics.
    """

    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:

        # Initialize the aggregated metrics
        aggregated_metrics = {k: 0.0 for k in metrics[0][1].keys()}

        # Aggregate the metrics
        for _, metric in metrics:
            for k, v in metric.items():
                aggregated_metrics[k] += v

        for k in aggregated_metrics.keys():
            aggregated_metrics[k] /= len(metrics)

        return aggregated_metrics

    return fit_metrics_aggregation_fn


def get_evaluate_fn(
    model: torch.nn.Module,
    s_model: torch.nn.Module,
    head: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    data_logger: Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]]] = None,
    eval_every: Optional[int] = 1,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
    """
    Get the function to evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        s_model (torch.nn.Module): The server model.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device on which the evaluation will be performed.
        dataloader (torch.utils.data.DataLoader): The test data loader.
        data_logger (Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]]], optional):
            The data logger.

    Returns:
        Callable[int, NDArrays, Dict[str, Scalar], Tuple[float, Dict[str, Scalar]]]:
            The function to evaluate the model on the test set.
    """

    # Parse the device
    if isinstance(device, str):
        device = torch.device(device)

    # Define the evaluation function
    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],  # pylint: disable=unused-argument
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        if "force_eval" not in config or not config["force_eval"]:
            if eval_every > 1 and (server_round == 0 or server_round % eval_every != 0):
                return None

        # Update the model with the latest parameters
        models = OrderedDict({"model": model, "head": head})
        set_weights(models, parameters)

        # Add console logging
        log(INFO, "Evaluating the model...")

        # Evaluate the model
        metrics = _eval_fn(
            model=model,
            s_model=s_model,
            head=head,
            device=device,
            criterion=criterion,  # config["criterion"],
            dataloader=dataloader,
        )

        log(INFO, "Evaluation done.")

        # Log the losses and KD loss on tensorboard or MLFlow
        if data_logger is not None:
            with (
                mlflow.start_run(
                    run_id=data_logger["mlflow"]["run_id"],
                    run_name=data_logger["mlflow"]["run_name"],
                    log_system_metrics=False,
                )
                if list(data_logger.keys())[0] == "mlflow"
                else nullcontext()
            ):
                data = {}

                data["Federated/Eval Server accuracy-1"] = metrics["s_acc_1"] * 100
                data["Federated/Eval Server accuracy-5"] = metrics["s_acc_5"] * 100
                data["Federated/Eval Server loss"] = metrics["s_loss"]
                data["Federated/Eval Server ECE"] = metrics["s_ece"] * 100

                data["Federated/Eval Client accuracy-1"] = metrics["acc_1"] * 100
                data["Federated/Eval Client accuracy-5"] = metrics["acc_5"] * 100
                data["Federated/Eval Client loss"] = metrics["loss"]
                data["Federated/Eval Client ECE"] = metrics["ece"] * 100

                # Log the classifier and the self-similarity
                w = head.fcn[-1].weight.detach().cpu()

                # Plot the classifier weights
                fig, ax = plt.subplots()
                ax.imshow(w, cmap="viridis")
                fig.tight_layout()

                data["Federated/Classifier Weights"] = fig

                # Plot the self-similarity
                fig, ax = plt.subplots()
                w_norm = w / torch.norm(w, dim=1, keepdim=True)
                similarity = w_norm @ w_norm.T
                ax.imshow(similarity, cmap="viridis")
                fig.tight_layout()

                data["Federated/Self-Similarity"] = fig

                log2logger(
                    logger=data_logger,
                    epoch=(server_round),
                    data=data,
                )

        return metrics["mean_loss"], {
            "client_loss": metrics["loss"],
            "client_accuracy_1": metrics["acc_1"],
            "client_accuracy_5": metrics["acc_5"],
            "client_ece": metrics["ece"],
            "server_loss": metrics["s_loss"],
            "server_accuracy_1": metrics["s_acc_1"],
            "server_accuracy_5": metrics["s_acc_5"],
            "server_ece": metrics["s_ece"],
        }

    return evaluate_fn


def _eval_fn(
    model: torch.nn.Module,
    s_model: torch.nn.Module,
    head: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ECE_M: int = 15,  # pylint: disable=invalid-name
) -> Tuple[float, Dict[str, Scalar]]:
    """
    Evaluate both the server and client models on the test set.

    Args:
        model (torch.nn.Module): The client model.
        s_model (torch.nn.Module): The server model.
        device (Union[torch.device, str]): The device on which the evaluation will be performed.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The test data loader.
        ECE_M (int): Hyperparameter used to compute the ECE metric,
                        default is 15 as per original paper (https://arxiv.org/pdf/1706.04599).

    Returns:
        Tuple[float, Dict[str, Scalar]]: The loss and accuracy of the client model.
    """

    # Cast the device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Evaluate the model
    # Set all the models to eval mode
    set_eval_mode(model=model, head=head, s_model=s_model)

    # Initialize accumulation variables
    # To compute ECE we need the full set of conf/pred/label,
    # therefore these will have an entry for each sample
    labels_acc = []

    c_preds_acc = []
    c_confs_acc = []
    s_preds_acc = []
    s_confs_acc = []

    # The loss is aggregated batch-wise
    c_loss_acc = []
    s_loss_acc = []

    # Eval the model
    for i, batch in enumerate(dataloader):
        # Run the batch
        # Move the data to the device
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(device), labels.to(device)
        labels_acc.append(labels.detach())  # Accumulate labels

        # Compute the network prediction
        with torch.no_grad():
            feats = model(images)
            outputs = head(normalize_tensor(feats))
            if batch["embeddings"] is not None:
                s_feats = batch["embeddings"].to(device)
            else:
                s_feats = s_model.get_class_scores(image=images, normalize=True)
            s_outputs = head(normalize_tensor(s_feats))

        # Compute the loss
        loss = criterion(outputs, labels)
        s_loss = criterion(s_outputs, labels)

        # Accumulate the losses
        c_loss_acc.append(loss.detach())
        s_loss_acc.append(s_loss.detach())

        # Compute the softmax to extract the prediction confidence (i.e. probability)
        outputs = torch.softmax(outputs, dim=-1)
        s_outputs = torch.softmax(s_outputs, dim=-1)

        # Get the class predictions
        conf, top_k = torch.topk(outputs, k=5, dim=1)
        s_conf, s_top_k = torch.topk(s_outputs, k=5, dim=1)

        # Accumulate the prediction confidence
        c_confs_acc.append(conf.detach())
        s_confs_acc.append(s_conf.detach())

        # Accumulate the class predictions
        c_preds_acc.append(top_k.detach())
        s_preds_acc.append(s_top_k.detach())

        # Compute the batch accuracy
        acc_1 = torch.mean(1.0 * (top_k[:, 0] == labels)).item()
        s_acc_1 = torch.mean(1.0 * (s_top_k[:, 0] == labels)).item()

        # Log the progress
        if i % 10 == 0:
            log(
                INFO,
                "[%03d/%03d] Loss: [server=%.4f, client=%.4f] | "
                + "Accuracy: [server=%.2f%% , client=%.2f%%]",
                i,
                len(dataloader),
                s_loss,
                loss,
                s_acc_1 * 100,
                acc_1 * 100,
            )

    # Flatten the accumulation vectors
    labels_acc = torch.cat(labels_acc, dim=0)  # shape: (N,)

    c_preds_acc = torch.cat(c_preds_acc, dim=0)  # shape: (N,5)
    c_confs_acc = torch.cat(c_confs_acc, dim=0)  # shape: (N,5)
    s_preds_acc = torch.cat(s_preds_acc, dim=0)  # shape: (N,5)
    s_confs_acc = torch.cat(s_confs_acc, dim=0)  # shape: (N,5)

    c_loss_acc = torch.tensor(c_loss_acc)  # shape: (Nb,)
    s_loss_acc = torch.tensor(s_loss_acc)  # shape: (Nb,)

    # Extract the total loss and accuracy
    acc_1 = torch.mean(1.0 * (c_preds_acc[:, 0] == labels_acc)).item()
    acc_5 = torch.mean(
        1.0 * torch.any(c_preds_acc == labels_acc.unsqueeze(1), dim=1)
    ).item()
    s_acc_1 = torch.mean(1.0 * (s_preds_acc[:, 0] == labels_acc)).item()
    s_acc_5 = torch.mean(
        1.0 * torch.any(s_preds_acc == labels_acc.unsqueeze(1), dim=1)
    ).item()

    loss = c_loss_acc.mean().item()
    s_loss = s_loss_acc.mean().item()
    mean_loss = (loss + s_loss) / 2.0

    # Compute the ECE metric
    c_ECE = 0  # pylint: disable=invalid-name
    s_ECE = 0  # pylint: disable=invalid-name

    # Extract prediction confidence
    c_p = c_confs_acc[:, 0]
    s_p = s_confs_acc[:, 0]

    # Sample i is in Bm <=> (m-1)/M < p_i <= m/M, where m = 1...M
    for m in range(1, ECE_M + 1):
        # Compute the Bm sets
        c_Bm = torch.where(  # pylint: disable=invalid-name
            torch.logical_and(c_p > (m - 1) / ECE_M, c_p <= m / ECE_M)
        )[
            0
        ]  # output is a tuple of tensors, extract the only valid element
        s_Bm = torch.where(  # pylint: disable=invalid-name
            torch.logical_and(s_p > (m - 1) / ECE_M, s_p <= m / ECE_M)
        )[
            0
        ]  # output is a tuple of tensors, extract the only valid element

        # Compute client Bm metrics
        if torch.numel(c_Bm) > 0:
            c_acc_Bm = torch.mean(  # pylint: disable=invalid-name
                1.0 * (c_preds_acc[c_Bm, 0] == labels_acc[c_Bm])
            )
            c_conf_Bm = torch.mean(c_p[c_Bm])  # pylint: disable=invalid-name
            c_ECE += (  # pylint: disable=invalid-name
                c_Bm.shape[0] * torch.abs(c_acc_Bm - c_conf_Bm).item() / c_p.shape[0]
            )

        # Compute server Bm metrics
        if torch.numel(s_Bm) > 0:  # pylint: disable=invalid-name
            s_conf_Bm = torch.mean(s_p[s_Bm])  # pylint: disable=invalid-name
            s_acc_Bm = torch.mean(  # pylint: disable=invalid-name
                1.0 * (s_preds_acc[s_Bm, 0] == labels_acc[s_Bm])
            )
            s_ECE += (  # pylint: disable=invalid-name
                s_Bm.shape[0] * torch.abs(s_acc_Bm - s_conf_Bm).item() / s_p.shape[0]
            )

    # Define the output metrics
    metrics = {
        "mean_loss": mean_loss,
        "s_loss": s_loss,
        "loss": loss,
        "s_acc_1": s_acc_1,
        "s_acc_5": s_acc_5,
        "acc_1": acc_1,
        "acc_5": acc_5,
        "ece": c_ECE,
        "s_ece": s_ECE,
    }

    # Set all the models to train mode
    set_train_mode(model=model, head=head, s_model=s_model, module2train="all")

    return metrics
