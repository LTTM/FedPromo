"""
Implementation of the FedAlign strategy.
The FedAlign strategy is a federated learning strategy that aims
to align the local models of the clients to the global model.

On the server aggration step, the global model is fine-tuned using
a subset of the in-domain pre-training samples.

Due to class miss-alignment the fine-tune is performed unsupervised.

Classes:
    FedAlign: The FedAlign strategy.


Functions:
    None

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Optional, Union, Callable, Dict
from logging import INFO

import torch
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    MetricsAggregationFn,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.common.typing import NDArrays
from omegaconf import DictConfig
from hydra.utils import instantiate

from modules.common.training_utils import set_optimizer
from modules.common.training_utils import set_train_mode, set_eval_mode
from modules.common.data_types import TorchModule, TorchOptimObj
from modules.federated.clients.utils import get_weights, set_weights
from modules.datasets.data_handler import load_dts


# Define the default export
__all__ = ["FedAlign", "get_align_fn"]


class FedAlign(FedAvg):
    """
    The FedAlign strategy.

    Args:
        model(TorchModule): The model to train.
        s_model(TorchModule): The source model.
        head(TorchModule): The head of the model.
        optimizer(TorchOptimObj): The optimizer to use for the training.
        generator(torch.Generator): The generator to use for the alignment.
        align_fn (Callable[[Parameters], Parameters]): The function used to align the model
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]
            ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        inplace : bool (default: True)
            Enable (True) or disable (False) in-place aggregation of model updates.
    """

    def __init__(
        self,
        *,
        model: TorchModule,
        s_model: TorchModule,
        head: TorchModule,
        optimizer: TorchOptimObj,
        generator: Optional[torch.Generator] = None,
        align_fn: Callable[[Parameters], Parameters],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        # Call the super class
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

        # Define the modules
        self.model = model
        self.s_model = s_model
        self.head = head

        # Defien the optimizer
        self.optimizer = optimizer

        # Define the generator
        self.generator = generator

        # Define the align data
        self.align_data = None

        # Define the align function
        self.align_fn = align_fn

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregate fit results using weighted average.

        Args:
            server_round (int): The current server round.
            results (list[tuple[ClientProxy, FitRes]]): The results of the clients.
            failures (list[Union[tuple[ClientProxy, FitRes], BaseException]]):
                The failures of the clients.

        Returns:
            tuple[Optional[Parameters], dict[str, Scalar]]: The aggregated parameters and metrics.
        """

        # Call the super class
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Perform the alignment
        log(INFO, "Starting alignment")
        parameters_aggregated, losses, self.align_data = self.align_fn(
            model=self.model,
            s_model=self.s_model,
            head=self.head,
            optimizer=self.optimizer,
            generator=self.generator,
            parameters=parameters_aggregated,
            align_data=self.align_data,
        )
        log(INFO, "Alignment completed, average loss: %.4e", losses.mean())

        # Append the loss on the metrics
        metrics_aggregated["fedalign_average_round_loss"] = losses.mean()

        return parameters_aggregated, metrics_aggregated


def get_align_fn(
    criterion: Union[DictConfig, TorchModule],
    align_dts_info: DictConfig,
    batch_size: int,
    data_folder: str,
    lr: Optional[float] = None,
    transforms: Optional[DictConfig] = None,
    dataloader_workers: Optional[int] = 0,
    module2train: Optional[str] = "head",
    log_every: Optional[int] = None,
    max_iters: Optional[int] = None,
) -> Callable[[Parameters], Parameters]:
    """
    Instantiate the align function.

    Args:
        criterion (Union[DictConfig, TorchModule]): The loss function.
        align_dts_info (DictConfig): The information about the alignment dataset.
        batch_size (int): The batch size to use for the alignment.
        data_folder (str): The folder where the data is stored.
        seed (Optional[int], optional): The seed to use for the alignment. Defaults to 42.
        generator (Optional[torch.Generator], optional): The generator to use for the alignment.
        lr (Optional[float], optional): The learning rate to use for the alignment.
            Defaults to None.
        transforms (Optional[DictConfig], optional): The transforms to use for the alignment.
            Defaults to None.
        dataloader_workers (Optional[int], optional): The number of workers to use
            for the dataloader.
            Defaults to 0.
        module2train (Optional[str], optional): The module to train. Defaults to "head".
        log_every (Optional[int], optional): The frequency to log the loss. Defaults to None.
        max_iters (Optional[int], optional): The maximum number of iterations. Defaults to None.

    Returns:
        Callable[[Parameters], Parameters]: The align function.
    """

    # Instantiate the criterion
    if isinstance(criterion, DictConfig):
        criterion = instantiate(criterion)

    # Instantiate the transforms
    if transforms is not None and isinstance(transforms, DictConfig):
        transforms = instantiate(transforms)
    if not isinstance(transforms, Dict):
        transforms = {"train": transforms, "val": transforms}

    def align_fn(
        model: TorchModule,
        s_model: TorchModule,
        head: TorchModule,
        optimizer: TorchOptimObj,
        parameters: Parameters,
        generator: Optional[torch.Generator] = None,
        align_data: Optional[torch.utils.data.DataLoader] = None,
    ) -> Parameters:
        """
        Perform the allignment.

        Args:
            model (TorchModule): The model to align.
            s_model (TorchModule): The source model.
            head (TorchModule): The head of the model.
            optimizer (TorchOptimObj): The optimizer to use for the alignment.
            parameters (Parameters): The parameters of the model.
            generator (Optional[torch.Generator], optional): The generator to use for the alignment.
                Defaults to None.

        Returns:
            tuple[Optional[Parameters], dict[str, Scalar]]: The aggregated parameters and metrics.
        """

        # Defien the generator if not provided
        if generator is None:
            generator = torch.Generator().manual_seed(42)

        # Define the device
        m_device = next(model.parameters()).device

        # Load the alignment data
        if align_data is None:
            align_data = load_dts(
                dts_info=align_dts_info,
                batch_size=batch_size,
                cache_dir=data_folder,
                teacher_name=s_model.__class__.__name__.lower(),
                transforms=transforms,
                generator=generator,
                num_workers=dataloader_workers,
            )[0]["train"]

        # Initialize the max_iters var
        if max_iters is None:
            m_iters = len(align_data)
        else:
            m_iters = min(max_iters, len(align_data))

        # Set the parameters
        set_weights({"model": model, "head": head}, parameters_to_ndarrays(parameters))

        # Set the correct module in training mode
        set_train_mode(model=model, head=head, module2train=module2train)

        # Instantiate the optimizer
        optim, _ = set_optimizer(
            model=model,
            head=head,
            optimizer=optimizer,
            lr=lr,
            module2train=module2train,
        )

        # Perform the alignment
        losses = torch.zeros(len(align_data))
        for index, batch in enumerate(align_data):
            # Check if the max iterations is reached
            if index >= m_iters:
                break

            # Zero the gradients
            optim.zero_grad()

            # Get the data
            images = batch["image"].to(m_device)
            if batch["embeddings"] is not None:
                s_feats = batch["embeddings"].to(m_device)
            else:
                s_feats = s_model.get_class_scores(image=images, normalize=True)

            # Perform the forward pass
            c_out = head(model(images))
            s_out = head(s_feats)

            # Compute the loss
            loss = criterion(c_out, s_out)
            losses[index] = loss.item()

            # Log the loss
            if log_every is not None and index % log_every == 0:
                log(
                    INFO,
                    "[Batch %03d/%03d] " + "Batch loss: %.4e",
                    index + 1,
                    m_iters,
                    loss.item(),
                )

            # Perform the backward pass
            loss.backward()

            # Perform the optimization step
            optim.step()

        # Get the parameters
        parameters = ndarrays_to_parameters(get_weights({"model": model, "head": head}))

        # Set the model in evaluation mode
        set_eval_mode(model=model, head=head, s_model=s_model)

        return parameters, losses, align_data

    return align_fn
