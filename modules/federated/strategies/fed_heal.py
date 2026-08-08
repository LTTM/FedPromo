"""
Implementation of the FedHEAL strategy.
The FedHEAL strategy is a federated learning strategy that perform
a smart average of the weight to increase aggregation performance.

Classes:
    FedHEAL: The FedHEAL strategy.

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
__all__ = ["FedHEAL"]


class FedHEAL(FedAvg):
    """
    The FedHEAL strategy.

    Args:
        beta: Optional[float], optional
            Hyperparameter controlling the FedHEAL re-weighting strenght.
            beta=0 ==> FedAvg, beta==1 ==> FedHEAL only. Defaults to 0.4.
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
        beta: Optional[float] = 0.4,
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
        self.beta = beta
        self.prev_model = None

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
        params = [[weights_results[c][0][p] for c in range(len(weights_results))] for p in range(len(weights_results[0][0]))]

        tot_examples = sum(r for _, r in weights_results)
        fedavg_weights = [r/tot_examples for _, r in weights_results]

        if self.prev_model is None: # Fallback to FedAVG
            aggregated_ndarrays = [sum(p*w for p,w in zip(ps, fedavg_weights)) for ps in params]
        else:
            diffs = np.array([np.mean([np.mean((oldp - p)**2) for oldp, p in zip(self.prev_model, ps)]) for ps, _ in weights_results])
            diffs /= diffs.sum() + 1e-8
            fedheal_weights = [(1-self.beta)*avgw + self.beta*healw for avgw, healw in zip(fedavg_weights, diffs)]
            aggregated_ndarrays = [sum(p*w for p,w in zip(ps, fedheal_weights)) for ps in params]

        self.prev_model = aggregated_ndarrays.copy() # save server params for next round
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
