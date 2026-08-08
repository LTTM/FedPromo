"""
Module with auxiliary functions.

Classes:
    TimeEstimator: Class to estimate the remaining time of a process.
    ParseKwargs: Class to parse the keyword arguments.
    PreCompose: Class to compose the transformations using two different tr pipeline.

Functions:
    conf2compose: Convert the configuration to a transforms.v2.Compose object.
    conf2precompose: Convert the configuration to a torch.nn.Module.
    seconds2h_format: Convert seconds to hours, minutes and seconds.
    interpolation_mode: Convert the string to an InterpolationMode object.
    dtypes: Convert the string to a torch.dtype object.
    parse_def_conf: Parse the default configuration.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

from time import time
from typing import Optional, OrderedDict, Union, Dict, List, Any
from inspect import signature
from collections.abc import MutableMapping

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError
import torch
import numpy as np
from torchvision.transforms.v2 import Compose, Transform
from torchvision.transforms.v2.functional import InterpolationMode

from modules.common.logger import Logger


__all__ = [
    "TimeEstimator",
    "ParseKwargs",
    "conf2compose",
    "conf2precompose",
    "seconds2h_format",
    "interpolation_mode",
    "dtypes",
    "parse_def_conf",
    "filtered_instantiate",
    "safe_ckpt_load",
    "flatten_conf_dict",
]


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


class TimeEstimator:
    """
    Class to estimate the remaining time of a process.

    Args:
        start_time (float): The starting time.
        total (int): The total number of iterations.
        current (int): The current iteration.
    """

    def __init__(self, total: int) -> None:
        """
        Initialize the TimeEstimator object.

        Args:
            total (int): The total number of iterations.
        """

        self.start_time = 0
        self.prev_time = 0
        self.total = total
        self.current = 0
        self.time_log = np.zeros(total)

    def start(self) -> None:
        """
        Start the timer and reset.
        """

        # Start the timer
        self.start_time = time()

        # Reset
        self.current = 0
        self.time_log = np.zeros(self.total)
        self.prev_time = self.start_time

    def update(self) -> None:
        """
        Update the current iteration.

        Args:
            None
        Returns:
            None
        """

        # End the timer
        end_time = time()

        # Log the batch time
        tot_time = end_time - self.prev_time
        self.time_log[self.current] = tot_time

        # Update the previous time
        self.prev_time = end_time

        # Update the current iteration
        self.current += 1

    def get_last(self, h_format: Optional[bool] = False) -> Union[float, str]:
        """
        Get the time of the last iteration.

        Args:
            h_format (Optional[bool], optional): Whether to format the time. Defaults to False.
        Returns:
            Union[float, str]: The time of the last iteration.
        """

        # Define the time of the last iteration
        last = self.time_log[self.current - 1]

        if h_format:
            return seconds2h_format(last)
        else:
            return last

    def estimate(self, h_format: Optional[bool] = False) -> Union[float, str]:
        """
        Estimate the remaining time.

        Args:
            h_format (Optional[bool], optional): Whether to format the time. Defaults to False.

        Returns:
            Union[float, str]: The remaining time.
        """

        # Estimate the remaining time
        avg_time = np.mean(self.time_log[: self.current])
        rem_time = avg_time * (self.total - self.current)

        # Format the time
        if h_format:
            return seconds2h_format(rem_time)
        else:
            return rem_time

    def get_total(self, h_format: Optional[bool] = False) -> Union[float, str]:
        """
        Get the total time.

        Args:
            h_format (Optional[bool], optional): Whether to format the time. Defaults to False.
        Returns:
            Union[float, str]: The total time.
        """

        # Compute the total time
        tot_time = time() - self.start_time

        if h_format:
            return seconds2h_format(tot_time)
        else:
            return tot_time


class ParseKwargs:
    """
    Class to parse the keyword arguments.

    Args:
        **kwargs: The keyword arguments.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _PreCompose(torch.nn.Module):
    """
    Class to compose the transformations using two different tr pipelines.
    It is the same as standard Compose but accepts two different Compose pipelines
    and returns two different values.

    Args:
        tr_client (List[Transform]): The transformations to apply to the client.
        tr_server (List[Transform]): The transformations to apply to the server.
    """

    def __init__(self, tr_client: List[Transform], tr_server: List[Transform]) -> None:
        super().__init__()

        self.tr_client = Compose(tr_client)
        self.tr_server = Compose(tr_server)

    def forward(self, *inputs: Any) -> Dict[str, torch.Tensor]:
        """
        Apply the transformations to the input.

        Args:
            *inputs (Any): The input tensor.
        Returns:
            Dict[str, torch.Tensor]: The transformed tensor.
        """

        return {"client": self.tr_client(inputs), "server": self.tr_server(inputs)}


def conf2compose(**kwargs) -> Compose:
    """
    Convert the configuration to a transforms.v2.Compose object.

    Args:
        **kwargs: The transformations to apply.
    Returns:
        Compose: The composed transformations.
    """

    return Compose([v for v in kwargs.values()])


def conf2precompose(
    client_tr: Dict[str, torch.nn.Module], server_tr: Dict[str, torch.nn.Module]
) -> Compose:
    """
    Convert the configuration to a torch.nn.Module.
    The new Compose will return two different value one trasformed using the
    client_tr and the other using the server_tr.

    Args:
        client_tr (Dict[str, torch.nn.Module]): The transformations to apply to the client.
        server_tr (Dict[str, torch.nn.Module]): The transformations to apply to the server.
    Returns:
        Compose: The composed transformations.
    """

    return _PreCompose([v for v in client_tr.values()], [v for v in server_tr.values()])


def seconds2h_format(sec: float) -> str:
    """
    Convert seconds to hours, minutes and seconds

    Args:
        sec (float): seconds
    Returns:
        str: hours, minutes and seconds
    """

    sec = sec % (24 * 3600)
    hours = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60

    if sec < 1 and hours == 0 and minutes == 0:
        ms = sec * 1000
        return f"{ms:03.0f}ms"

    return f"{hours:02.0f}:{minutes:02.0f}:{sec:02.0f}"


def interpolation_mode(mode: str) -> InterpolationMode:
    """
    Convert the string to an InterpolationMode object.

    Args:
        mode (str): The interpolation mode.
    Returns:
        InterpolationMode: The interpolation mode object.
    """

    return getattr(InterpolationMode, mode.upper())


def dtypes(dtype: str) -> torch.dtype:
    """
    Convert the string to a torch.dtype object.

    Args:
        dtype (str): The data type.
    Returns:
        torch.dtype: The data type object.
    """

    return getattr(torch, dtype)


def parse_def_conf(cfg: DictConfig, name: str) -> Union[DictConfig, None]:
    """
    Parse the default configuration.
    If a default configuration is set to null or is not present return None.
    Otherwise return the default configuration.

    Args:
        cfg (DictConfig): The configuration object.
        name (str): The name of the configuration under evaluation.
    Returns:
        Union[DictConfig, None]: The default configuration.
    """

    try:
        return cfg[name]
    except ConfigKeyError:
        return None


def filtered_instantiate(obj: DictConfig, **kwargs) -> Any:
    """
    Extension of hydra.utils.instantiate that allows to filter the kwargs passed to the object
    in order to avoid passing unwanted parameters.

    Args:
        obj (DictConfig): The object to instantiate.
        **kwargs: The keyword arguments.
    Returns:
        Any: The instantiated object.
    """

    # Extract the object to be instantiated
    target_obj = obj._target_  # pylint: disable=protected-access

    # Load the object
    if "." in target_obj:
        # Extract the module and the class/function
        module, target = target_obj.rsplit(".", 1)

        # Import the module
        mod = __import__(module, fromlist=[target])

        # Get the target object
        target_obj = getattr(mod, target)
    else:
        # Get the target object
        target_obj = globals()[target_obj]

    # Get the signature of the object
    if hasattr(target_obj, "__init__"):
        accepted_params = signature(target_obj.__init__).parameters
    else:
        accepted_params = signature(target_obj).parameters

    # Append to the kwargs the kwarks in the DictConfig
    for k, v in obj.items():
        if k != "_target_":
            kwargs[k] = v

    # Filter the kwargs
    valid_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

    # Instantiate the object
    return target_obj(**valid_kwargs)


def safe_ckpt_load(
    model: torch.nn.Module,
    state_dict_to_load: OrderedDict[str, torch.Tensor],
    model_name_to_log: Optional[str] = None,
) -> None:
    """
    Load the checkpoint safely.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        state_dict_to_load (OrderedDict[str, torch.Tensor]): The state dict to load.
        model_name_to_log (Optional[str], optional): The name of the model to log. Defaults to None.
    Returns:
        None: The function modifies the model in place.
    """

    # Log the model name if provided
    if model_name_to_log is not None:
        logger.info("Loading checkpoint for model: %s", model_name_to_log)

    # Get the state_dict of the model
    odict = model.state_dict()

    # Check if the state_dict_to_load is None or empty
    if state_dict_to_load is None or len(state_dict_to_load) == 0:
        logger.warning("The state_dict_to_load is empty or None, skipping loading.")
        return

    # Loading the state_dict
    loaded_keys = 0
    for k, v in state_dict_to_load.items():
        if k in odict:
            if torch.all(torch.tensor(odict[k].shape) == torch.tensor(v.shape)):
                odict[k] = v
                loaded_keys += 1
            else:
                logger.warning(
                    "Ignoring key {%s} with shape %s, expected shape %s",
                    k,
                    v.shape,
                    odict[k].shape,
                )
        else:
            logger.warning("Ignoring key {%s}", k)

    # Log the number of keys loaded
    logger.info(
        "Loaded %d/%d keys from the state_dict of the model",
        loaded_keys,
        len(state_dict_to_load),
    )

    model.load_state_dict(odict, strict=True)


def flatten_conf_dict(
    dictionary: Union[Dict[str, Any], DictConfig], parent_key: str = "", separator="."
) -> Dict[str, Any]:
    """
    Flatten the configuration dictionary.

    Args:
        dictionary (Union[Dict[str, Any], DictConfig]): The configuration dictionary to flatten.
        parent_key (str, optional): The base key to use for the flattened keys. Defaults to "".
        separator (str, optional): The separator to use between keys.    Defaults to ".".
    Returns:
        Dict[str, Any]: The flattened configuration dictionary.
    """

    if isinstance(dictionary, DictConfig):
        # Convert DictConfig to a regular dictionary
        dictionary = OmegaConf.to_container(dictionary, resolve=True)

    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_conf_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
