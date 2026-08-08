"""
Module that contains the code to initialize the
logger and the console handler.

Classes:
    Logger: Logger class that allows
        to create a logger with a specific name and format.
    TqdmLogger: File-like
        class redirecting tqdm progress bar to given logging logger.
    MLFlowNameFilter: Formatter class that changes the logger name to "mlflow".

Functions:
    instantiate_mlflow: Instantiate the mlflow logging.

Constants:
    LOG_LEVELS: Dictionary that maps the log levels to their values.
    ACCEPTED_DATA_LOGGER: List of accepted data logger.

Exceptions:
    ValueError (Exception): Raised when the logger is not accepted.

Author: Matteo Caligiuri
"""

import logging
from typing import Optional, Dict, Union, Any
from matplotlib import pyplot as plt
from matplotlib import figure

import mlflow
from torch.utils.tensorboard import SummaryWriter

from modules.common.constants import ACCEPTED_DATA_LOGGER


__all__ = [
    "Logger",
    "TqdmLogger",
    "instantiate_mlflow",
    "log2logger",
    "ACCEPTED_DATA_LOGGER",
    "MLFlowNameFilter",
]


# Constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOGGER_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
LOGGER_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"


class Logger:
    """
    Logger class that allows to create a logger with a
    specific name and format.

    Args:
        name (str): The name of the logger.
        level (str): The level of the logger.
        f (Optional[str]): The format of the logger.
    """

    def __init__(self, name: str, level: str, f: Optional[str] = None) -> None:
        # Parse the name
        name = name.split(".")[-1]

        # Create the logger
        self.logger = logging.getLogger(name)

        # Set the logger level
        self.logger.setLevel(LOG_LEVELS[level.upper()])

        # Set the format
        self.set_format(f)

    def get_logger(self) -> logging.Logger:
        """Return the logger."""
        return self.logger

    def set_format(self, f: Optional[str] = None) -> None:
        """
        Set the format of the logger.

        Args:
            f (Optional[str]): The format of the logger.

        Returns:
            None
        """

        # Set the default format
        if f is None:
            f = LOGGER_FORMAT

        # Define the formatter
        formatter = logging.Formatter(fmt=f, datefmt=LOGGER_DATE_FORMAT)

        # Set the formatter
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)


class TqdmLogger:
    """
    File-like class redirecting tqdm progress bar to given logging logger.
    To use it, create an instance of this class and pass it to the
    `file` argument of the tqdm function.

    Args:
        logger (logging.Logger): The logger to redirect the tqdm progress bar.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def write(self, msg: str) -> None:
        """
        Use the logger to write the message.

        Args:
            msg (str): The message to write.
        """

        self.logger.info(msg.lstrip("\r"))

    def flush(self) -> None:
        """
        Do nothing.
        """


class MLFlowNameFilter(logging.Filter):
    """
    Formatter class that changes the logger name to 'mlflow'.
    """

    def filter(self, record):
        record.name = "mlflow"
        return True


def instantiate_mlflow(
    tracking_uri: str,
    experiment_name: str,
    experiment_description: Optional[str] = None,
    experiment_tags: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None,
) -> Optional[str]:
    """
    Instantiate the mlflow object.

    Args:
        tracking_uri (str): The uri of the mlflow server.
        experiment_name (str): The name of the experiment.
        experiment_description (Optional[str]): The description of the experiment.
        experiment_tags (Optional[Dict[str, str]]): The tags of the experiment.
        run_name (Optional[str]): The name of the run.

    Returns:
        Optional[str]: The name of the run.
    """

    # Set the experiment tags
    if experiment_tags is None:
        experiment_tags = {}

    # Set the tracking uri
    mlflow.set_tracking_uri(tracking_uri)

    # Check if the experiment already exists
    if mlflow.get_experiment_by_name(experiment_name) is None:
        # Create the experiment
        mlflow_client = mlflow.MlflowClient(tracking_uri=tracking_uri)

        experiment_tags["mlflow.note.content"] = experiment_description

        artifact_location = f"mlflow-artifacts:/{experiment_name}"

        mlflow_client.create_experiment(
            experiment_name, artifact_location=artifact_location, tags=experiment_tags
        )

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Return the run name (could be None)
    return run_name


def log2logger(
    logger: Dict[str, Any],
    epoch: int,
    data: Dict[str, Union[str, int, float, figure.Figure]],
) -> None:
    """
    Log the data to the logger.

    Args:
        logger (Dict[str, Any]): The logger to use.
        epoch (int): The current epoch.
        data (Dict[str, Union[str, int, float, matplotlib.figure.Figure]): The data to log.

    Returns:
        None
    """

    # Check if the logger is accepted
    if list(logger.keys())[0] not in ACCEPTED_DATA_LOGGER:
        raise ValueError(
            f"Logger {logger} not accepted. Choose one of {ACCEPTED_DATA_LOGGER}"
        )
    elif list(logger.keys())[0] == "tensorboard":
        # Check that the writer is present and is a SummaryWriter
        writer = logger["tensorboard"]
        if not isinstance(writer, SummaryWriter):
            raise ValueError("The writer must be a SummaryWriter.")

    logger = list(logger.keys())[0]

    # Define the logger function
    if logger == "mlflow":

        def log_metric_with_sync(*args, **kwargs):
            kwargs["synchronous"] = False
            mlflow.log_metric(*args, **kwargs)

        def log_figure_with_sync(*args, **kwargs):  # pylint: disable=unused-argument
            file_name = f"{args[0]}_{args[2]}.pdf"
            mlflow.log_figure(figure=args[1], artifact_file=file_name)

        log = log_metric_with_sync
        log_fig = log_figure_with_sync
    elif logger == "tensorboard":
        log = writer.add_scalar
        log_fig = writer.add_figure
    else:
        raise ValueError(
            f"Logger {logger} not accepted. Choose one of {ACCEPTED_DATA_LOGGER}"
        )

    # Log the data
    for key, value in data.items():
        if isinstance(value, figure.Figure):
            log_fig(key, value, epoch)
            plt.close(value)
        else:
            log(key, value, epoch)
