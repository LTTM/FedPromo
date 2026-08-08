# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Local DP modifier.

Class slightly modified from the original implementation of FLWR
to make it work with our custom client app.

Author: Matteo Caligiuri
        Francesco Barbato
"""


from logging import INFO
from typing import Optional

import numpy as np
from flwr.client.typing import ClientAppCallable
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message

from modules.federated.common_dp import (
    add_localdp_gaussian_noise_to_params,
    compute_clip_model_update,
)


# Define what to export
__all__ = ["LocalDpMod"]


class LocalDpMod:
    """Modifier for local differential privacy.

    This mod clips the client model updates and
    adds noise to the params before sending them to the server.

    It operates on messages of type `MessageType.TRAIN`.

    Parameters
    ----------
    clipping_norm : float
        The value of the clipping norm.
    epsilon : float
        The privacy budget.
        Smaller value of epsilon indicates a higher level of privacy protection.
    delta : float
        The failure probability.
        The probability that the privacy mechanism
        fails to provide the desired level of privacy.
        A smaller value of delta indicates a stricter privacy guarantee.
    clipping_norm_scheduler : Optional[bool]
        If True, the clipping norm is rescaled by the learning rate scheduler scale.
        If False, the clipping norm is fixed.
    scheduler_max_clipping_norm : Optional[float]
        The maximum clipping norm to use when the clipping norm scheduler is enabled.
        If the rescaled clipping norm exceeds this value, it will be capped to this value.
    min_Ds_size : Optional[int]
        The minimum size of the dataset used to compute the sensitivity.
        If set to 0, the sensitivity is computed using hard bounds.
        If set to a positive value, the sensitivity is computed using soft bounds.
    module2train : str
        The module to train. Can be "all", "main" or "head".
    n_head_layers : int
        The number of head layers in the model.
        Used only if module2train is set to "main" or "head".

    Examples
    --------
    Create an instance of the local DP mod and add it to the client-side mods:

    >>> local_dp_mod = LocalDpMod( ... )
    >>> app = fl.client.ClientApp(
    >>>     client_fn=client_fn, mods=[local_dp_mod]
    >>> )
    """

    def __init__(
        self,
        clipping_norm: float,
        epsilon: float,
        delta: float,
        clipping_norm_scheduler: Optional[bool] = False,
        scheduler_max_clipping_norm: Optional[float] = 0.0,
        min_Ds_size: Optional[int] = 0,  # pylint: disable=invalid-name
        module2train: str = "head",
        n_head_layers: int = 1,
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if scheduler_max_clipping_norm < 0:
            raise ValueError(
                "The scheduler_max_clipping_norm should be a non-negative value."
            )

        if min_Ds_size < 0:
            raise ValueError("The min_Ds_size should be a non-negative value.")

        if epsilon < 0:
            raise ValueError("Epsilon should be a non-negative value.")

        if delta < 0:
            raise ValueError("Delta should be a non-negative value.")

        self.clipping_norm = clipping_norm
        self.clipping_norm_scheduler = clipping_norm_scheduler
        self.scheduler_max_clipping_norm = scheduler_max_clipping_norm
        self.min_Ds_size = min_Ds_size  # pylint: disable=invalid-name
        self.epsilon = epsilon
        self.delta = delta
        self.module2train = module2train
        self.n_head_layers = n_head_layers
        if self.module2train not in ["all", "main", "head"]:
            raise ValueError(
                "The module2train should be one of 'all', 'main' or 'head'."
            )
        if (
            self.module2train == "main" or self.module2train == "head"
        ) and self.n_head_layers <= 0:
            raise ValueError(
                "The n_head_layers should be a positive value when "
                "module2train is set to 'head' or 'main'."
            )

    def __call__(
        self, msg: Message, ctxt: Context, call_next: ClientAppCallable
    ) -> Message:
        """Perform local DP on the client model parameters.

        Parameters
        ----------
        msg : Message
            The message received from the server.
        ctxt : Context
            The context of the client.
        call_next : ClientAppCallable
            The callable to call the next middleware in the chain.

        Returns
        -------
        Message
            The modified message to be sent back to the server.
        """
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        fit_ins = compat.recordset_to_fitins(msg.content, keep_input=True)
        server_to_client_params = parameters_to_ndarrays(fit_ins.parameters)

        # Extract the clipping_norm sceduler if it exists
        fit_config = msg.content.configs_records["fitins.config"]
        scheduler_scale = (
            fit_config["scheduler_scale"] if "scheduler_scale" in fit_config else None
        )

        # If the scale is present rescale the clipping norm
        if scheduler_scale is not None:
            cur_clipping_norm = float(self.clipping_norm * scheduler_scale)
            if self.scheduler_max_clipping_norm > 0:
                cur_clipping_norm = min(
                    self.scheduler_max_clipping_norm, cur_clipping_norm
                )
            log(
                INFO,
                "LocalDpMod: clipping norm rescaled by scheduler scale %.4f to %.4f.",
                scheduler_scale,
                cur_clipping_norm,
            )
        else:
            cur_clipping_norm = self.clipping_norm

        # Compute the sensitivity
        # If min_Ds_size is set to 0, use hard bounds
        # If min_Ds_size is set to a positive value, use soft bounds
        if self.min_Ds_size > 0:
            cur_sensitivity = 2 * cur_clipping_norm / self.min_Ds_size
        else:
            cur_sensitivity = cur_clipping_norm

        # Call inner app
        out_msg = call_next(msg, ctxt)

        # Check if the msg has error
        if out_msg.has_error():
            return out_msg

        fit_res = compat.recordset_to_fitres(out_msg.content, keep_input=True)

        client_to_server_params = parameters_to_ndarrays(fit_res.parameters)

        # Clip the client update
        compute_clip_model_update(
            client_to_server_params,
            server_to_client_params,
            cur_clipping_norm,
            self.module2train,
            self.n_head_layers,
        )
        log(
            INFO,
            "LocalDpMod: parameters are clipped by value: %.4f.",
            cur_clipping_norm,
        )

        fit_res.parameters = ndarrays_to_parameters(client_to_server_params)

        # Add noise to model params
        fit_res.parameters = add_localdp_gaussian_noise_to_params(
            fit_res.parameters,
            cur_sensitivity,
            self.epsilon,
            self.delta,
            self.module2train,
            self.n_head_layers,
        )

        noise_value_sd = (
            cur_sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        )
        log(
            INFO,
            "LocalDpMod: local DP noise with %.4f stedv added to parameters",
            noise_value_sd,
        )

        # Append to the metrics the current clipping norm, sensitivity and noise value
        out_msg.content.configs_records["fitres.metrics"][
            "local_dp_clipping_norm"
        ] = cur_clipping_norm
        out_msg.content.configs_records["fitres.metrics"][
            "local_dp_sensitivity"
        ] = cur_sensitivity
        out_msg.content.configs_records["fitres.metrics"][
            "local_dp_noise_value_sd"
        ] = noise_value_sd

        out_msg.content = compat.fitres_to_recordset(fit_res, keep_input=True)
        return out_msg
