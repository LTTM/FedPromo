"""
Implementation of the FedAlign strategy.
The FedWeight strategy is a federated learning strategy that perform
a smart avarege of the weight to increase aggregation performance.

Classes:
    FedWeight: The FedWeight strategy.

Functions:
    None

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Optional, Union, Callable
from logging import WARNING

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    MetricsAggregationFn,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.common.typing import NDArrays
from flwr.server.strategy.aggregate import aggregate

import numpy as np


# Define the default export
__all__ = ["FedHPA"]


class FedHPA(FedAvg):
    """
    The FedHPA (High Pass Aggregation) strategy.

    Args:
        aggregation_weight : Optional[float], optional
            The weight of the aggregation. The weight is applied to the weighted
            average of the weights. On the FedAvg average is applied the
            (1 - aggregation_weight). Defaults to 0.5.
        p_client_dropout : float, optional
            Probability of client dropout. Defaults to 0.0.
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
        aggregation_weight: Optional[float] = 0.5,
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

        # Define aggregation weight
        self.aggregation_weight = aggregation_weight

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.

        Args:
            server_round (int): The current round number.
            parameters (Parameters): The current global model parameters.
            client_manager (ClientManager): The client manager.

        Returns:
            list[tuple[ClientProxy, FitIns]]: A list of tuples containing the client
                and the fit instruction.
        """

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = weighted_aggregate(
            results=weights_results, aggregation_weight=self.aggregation_weight
        )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def weighted_aggregate(
    results: list[tuple[NDArrays, int]], aggregation_weight: Optional[float] = 0.5
) -> NDArrays:
    """
    Compute weighted average.

    Args:
        results (list[tuple[NDArrays, int]]): List of tuples containing the weights
            and the number of examples.
        aggregation_weight (Optional[float]): The weight of the aggregation.
            The weight is applied to the weighted average of the weights.
            On the FedAvg average is applied the (1 - aggregation_weight).
            Defaults to 0.5.

    Returns:
        NDArrays: The weighted average of the weights.
    """
    # Compute the average weights
    aggregated_ndarrays = aggregate(results)

    # Compute MSE of each client parameters with respect to the average parameters
    diff_ndarrays = [
        [
            (layer - layer_prime) ** 2
            for layer, layer_prime in zip(weights, aggregated_ndarrays)
        ]
        for weights, _ in results
    ]

    # Traspose the list of MSE to have a list of layers
    t_diff_ndarrays = [
        np.stack([ws[i] for ws in diff_ndarrays]) for i in range(len(diff_ndarrays[0]))
    ]

    # Compute the softmax of the MSE
    softmax_diff_ndarrays = [
        np.exp(layer) / np.sum(np.exp(layer), axis=0, keepdims=True)
        for layer in t_diff_ndarrays
    ]

    # Weight the aggreghation with the softmax of the MSE
    weighted_weights = [
        sum([soft[c] * results[c][0][w] for c in range(len(results))])
        for w, soft in enumerate(softmax_diff_ndarrays)
    ]

    # Avarage the balanced weight with the FedAvg aggregation
    weights_prime = [
        ((1 - aggregation_weight) * layer) + (aggregation_weight * layer_prime)
        for layer, layer_prime in zip(aggregated_ndarrays, weighted_weights)
    ]

    return weights_prime
