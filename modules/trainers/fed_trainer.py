"""
This module contains the implementation of the FedTrainer class.
The FedTrainer class is responsible for training a federated learning model.

Classes:
    FedTrainer: A class responsible for training a federated learning model.

Functions:
    None

Constants:
    None

Exceptions:
    ValueError: Raised when an invalid model type is provided.
    ValueError: Raised when an invalid checkpoint type is provided.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union, Optional, Tuple, Dict, OrderedDict as OrderedDictType, Type
from collections import OrderedDict
from pathlib import Path
from contextlib import nullcontext
from copy import deepcopy as copy

import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from hydra.errors import InstantiationException
from torchvision.transforms.v2 import Compose
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.server.strategy import Strategy
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters
from flwr.client import NumPyClient
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import mlflow

from modules.datasets.data_handler import load_fed_dts, load_dts
from modules.common.logger import Logger
from modules.common.constants import ALLOWED_MODULE2TRAIN, ACCEPTED_DATA_LOGGER
from modules.common.decorators import filter_kwargs
from modules.common.auxiliary_fn import safe_ckpt_load
from modules.federated.clients.base_client_app import generate_client_fn
from modules.federated.server_app import generate_server_fn
from modules.federated.clients.utils import get_weights
from modules.federated.visualization import plot_label_distributions
from modules.federated.strategies.utils import set_strategy, get_evaluate_fn
from modules.models import SimpleClassifier


# Add the eval resolver to OmegaConf
OmegaConf.register_new_resolver("eval", eval)


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


# Define the element to export
__all__ = ["FedTrainer"]


class FedTrainer:
    """
    A class responsible for training a federated learning model.

    Args:
        c_model (Union[DictConfig, torch.nn.Module]): The classification model to use.
        s_model (Union[DictConfig, torch.nn.Module]): The segmentation model to use.
        n_classes (int): The number of classes in the dataset.
        criterion (Union[DictConfig, torch.nn.Module]): The criterion to use for the training.
        optimizer (Union[DictConfig, torch.optim.Optimizer]): The optimizer to use for the training.
        head (Optional[Union[DictConfig, torch.nn.Module]]): The head to use for the model.
            Defaults to None.
        device (Optional[Union[str, torch.device]]): The device to use for the training.
            Defaults to "cpu".
        data_folder (Optional[Union[Path, str]]): The folder where the data is stored.
            Defaults to Path("data").
        save_path (Optional[Union[Path, str]]): The path where to save the outputs.
            Defaults to Path("fed_outputs").
        checkpoint (Optional[Union[Path, str, OrderedDict]]): The checkpoint to load.
            Defaults to None.
        data_logger (Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]]]):
            The data logger to use.
            Defaults to None.
        dataloader_workers (Optional[int]): The number of workers to use for the dataloaders.

    Returns:
        None
    """

    def __init__(
        self,
        c_model: Union[DictConfig, torch.nn.Module],
        s_model: Union[DictConfig, torch.nn.Module],
        n_classes: int,
        criterion: Union[DictConfig, torch.nn.Module],
        optimizer: Union[DictConfig, torch.optim.Optimizer],
        head: Optional[Union[DictConfig, torch.nn.Module]] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        data_folder: Optional[Union[Path, str]] = Path("data"),
        save_path: Optional[Union[Path, str]] = Path("fed_outputs"),
        checkpoint: Optional[Union[Path, str, OrderedDictType]] = None,
        data_logger: Optional[
            Dict[str, Union[SummaryWriter, Dict[str, str], None]]
        ] = None,
        dataloader_workers: Optional[int] = 4,
    ) -> None:
        # Check the data folder and create it if it does not exist
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)
        if isinstance(save_path, str):
            save_path = Path(save_path)

        data_folder.mkdir(parents=True, exist_ok=True)

        self.data_folder = data_folder
        self.save_path = save_path

        # Set the device
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        # Set the models
        # Server model
        if isinstance(s_model, torch.nn.Module):
            self.s_model = s_model
        elif isinstance(s_model, DictConfig):
            self.s_model = instantiate(
                s_model,
                device=device,
                data_folder=data_folder,
                n_classes=n_classes,
            )
        else:
            raise ValueError(f"The server model type ({type(s_model)}) is not valid.")

        # Client model
        if isinstance(c_model, torch.nn.Module):
            self.c_model = c_model.to(device)
        elif isinstance(c_model, DictConfig):
            self.c_model = instantiate(
                c_model,
                out_feats_size=self.s_model.get_feats_shape(),
            ).to(device)
        else:
            raise ValueError(f"The client model type ({type(c_model)}) is not valid.")

        # Head
        if isinstance(head, torch.nn.Module):
            self.head = head.to(device)
        elif isinstance(head, DictConfig):
            self.head = instantiate(
                head,
                features=self.s_model.get_feats_shape(),
                num_classes=n_classes,
            ).to(device)
        else:
            self.head = SimpleClassifier(
                features=self.s_model.get_feats_shape(),
                num_classes=n_classes,
                num_layers=1,
                dropouts=None,
            ).to(device)

        # Handle the checkpoint
        self._load_checkpoint(checkpoint)

        # Instantiate the criterion
        # if they are DictConfig objects
        if isinstance(criterion, DictConfig):
            criterion = instantiate(criterion)

        # Set the criterion and the optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # Set the data logger
        if data_logger is None:
            self.data_logger = None
        elif (
            data_logger is not None
            and list(data_logger.keys())[0] in ACCEPTED_DATA_LOGGER
        ):
            self.data_logger = data_logger
        else:
            raise ValueError(
                f"Data logger {data_logger} not accepted. Choose one of {ACCEPTED_DATA_LOGGER}"
            )

        # Define the number of workers
        self.dataloader_workers = dataloader_workers

        # Initialize the client
        self.client = None

        # Initialize the server
        self.server = None

        # Initialize the dataloaders
        self.dataloaders = None
        self.full_test_dataloader = None

        # Initialize some optional variables used during logging
        self.dts_name = None
        self.n_clients = None
        self.n_clients_per_round = None
        self.n_epochs_per_client = None
        self.head_type = {"n_linear": head.num_layers, "dropouts": head.dropouts}

        # Initialize the generator
        self.generator = None

        # Initialize the evaluate function
        self.evaluate_fn = None

    def _load_checkpoint(self, checkpoint: Optional[DictConfig] = None) -> None:
        """
        Load the pretraining checkpoint.

        Args:
            checkpoint (Optional[DictConfig]): The checkpoint to load.

        Returns:
            None
        """

        # If checkpoint is None, return
        if checkpoint is None:
            logger.info(
                "No checkpoint provided. The client model will be initialized on "
                + "the ImageNet weights."
            )
            return

        # Extract path and module to load
        checkpoint_path = Path(checkpoint.path)
        checkpoint_module = checkpoint.module2load

        # Chech if the checkpoint module is one of the accepted values
        if checkpoint_module not in ALLOWED_MODULE2TRAIN:
            raise ValueError(
                f"Checkpoint module {checkpoint_module} not accepted."
                + f"Choose one of {ALLOWED_MODULE2TRAIN}"
            )

        # Load the checkpoint
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        if "head" not in checkpoint_data:
            checkpoint_data["head"] = None

        # Load the model
        if checkpoint_module == "all":
            safe_ckpt_load(self.c_model, checkpoint_data["model"], "main")
            safe_ckpt_load(self.head, checkpoint_data["head"], "head")
        elif checkpoint_module == "main":
            safe_ckpt_load(self.c_model, checkpoint_data["model"], "main")
        elif checkpoint_module == "head":
            safe_ckpt_load(self.head, checkpoint_data["head"], "head")

    def set_data(
        self,
        dataset_info: DictConfig,
        batch_size: int,
        partitioner: DictConfig,
        transforms: Optional[Union[Compose, DictConfig]] = None,
        seed: Optional[int] = 42,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Set the data to use for the federated learning training and evaluation.

        Args:
            dataset_info (DictConfig): The dataset information.
            batch_size (int): The batch size to use.
            partitioner (DictConfig): The partitioner to use.
            transforms (Optional[Union[Compose, DictConfig]]): The transforms to apply to the data.
                Defaults to None.
            seed (Optional[int]): The seed to use for the data.
                Defaults to 42.
            generator (Optional[torch.Generator]): The generator to use for the data.
                If the generator is not provided, the seed will be used.
                Defaults to None.

        Returns:
            None
        """

        # Check if the transforms are provided
        # and parse them from DictConfig to Compose
        # if necessary
        if transforms is not None and isinstance(transforms, DictConfig):
            transforms = instantiate(transforms)

        # Define the genearator if the generator is not provided
        # but the seed is provided
        if generator is None and seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        elif generator is not None:
            self.generator = generator
        else:
            raise ValueError("Either the seed or the generator must be provided.")

        # Load the dataloaders of the dataset
        logger.info("Loading cliets dataset...")
        self.dataloaders = load_fed_dts(
            dts_info=dataset_info,
            batch_size=batch_size,
            transforms=transforms,
            seed=seed,
            generator=self.generator,
            partitioner=partitioner,
            num_workers=self.dataloader_workers,
        )
        logger.info("Dataset loaded.")
        logger.info("Loading the evaluation dataset...")
        test_dataloader = load_dts(
            dts_info=dataset_info,
            batch_size=batch_size,
            cache_dir=self.data_folder,
            teacher_name=self.s_model.__class__.__name__.lower(),
            transforms={"train": transforms, "val": transforms},
            generator=self.generator,
            num_workers=self.dataloader_workers,
        )
        try:
            self.full_test_dataloader = test_dataloader[0]["test"]
        except KeyError:
            self.full_test_dataloader = test_dataloader[0]["val"]
        del test_dataloader
        logger.info("Evaluation dataset loaded.")

        # Create all the partiotions
        # to speed up the training
        logger.info("Creating partitions...")
        for k in self.dataloaders.partitioners:
            self.dataloaders.partitioners[k].create_partitions()
        logger.info("Partitions created.")

        # Update dts name for logging
        self.dts_name = dataset_info.name

    def set_client_and_server(
        self,
        client_type: Union[DictConfig, Type[NumPyClient]],
        strategy: Union[DictConfig, Strategy],
        num_rounds: int,
        class_debias: Optional[bool] = False,
        class_debias_weight: Optional[float] = 1.0,
        log_every: Optional[int] = -1,
        eval_every: Optional[int] = 1,
        client_dp: Optional[DictConfig] = None,
        server_dp: Optional[DictConfig] = None,
    ) -> None:
        """
        Set the client and the server for the federated learning process.

        Args:
            client_type (Union[DictConfig, Type[NumPyClient]]): The type of client to use.
            strategy (Union[DictConfig, Strategy]): The strategy to use for the
                federated learning process
            num_rounds (int): The number of rounds to run the federated learning process.
            class_debias (Optional[bool]): Whether to use class debiasing.
            class_debias_weight (Optional[float]): Weight used in class debiasing.
            log_every (Optional[int]): The number of rounds after which to log the results.
                Defaults to -1.
            eval_every (Optional[int]): The number of rounds after which to evaluate the model.
                Defaults to 1.
            client_dp (Optional[DictConfig]): The differential privacy client configuration.
                Defaults to None.
            server_dp (Optional[DictConfig]): The differential privacy server configuration.
                Defaults to None.

        Returns:
            None
        """

        # Parse the client_type
        if isinstance(client_type, DictConfig):
            client_type = instantiate(client_type, _partial_=True)
        elif not issubclass(client_type, NumPyClient):
            raise ValueError(
                f"Client type {type(client_type)} not accepted. Choose one of {NumPyClient}"
            )

        # Generate the client_fn function
        client_fn = generate_client_fn(
            client_type=client_type,
            model=self.c_model,
            head=self.head,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
            dataloaders=self.dataloaders,
            save_path=self.save_path,
        )

        # Define the local differential privacy client mod if required
        if client_dp is not None:
            # Check if the dp is a DictConfig
            if not isinstance(client_dp, DictConfig):
                raise ValueError(
                    f"DP type {type(client_dp)} not accepted. Choose one of {DictConfig}"
                )

            # Add DP to the previously defined client_fn
            client_dp_mod = instantiate(client_dp)

            # Set the client with dp
            self.client = ClientApp(client_fn=client_fn, mods=[client_dp_mod])
        else:
            # Set the client without dp
            self.client = ClientApp(client_fn=client_fn)

        # Initialize the strategy if it is a DictConfig
        if isinstance(strategy, DictConfig):
            # Get the weights of the model
            models = OrderedDict({"model": self.c_model, "head": self.head})
            params = get_weights(models)

            # Convert to dict the DictConfig
            # This is necessary to instantiate only the child elements
            strategy = dict(strategy)

            # Update var for logging
            self.n_clients = strategy["min_available_clients"]
            self.n_clients_per_round = strategy["min_fit_clients"]
            self.n_epochs_per_client = strategy["on_fit_config_fn"]["config"][
                "local_epochs"
            ]

            # Define the save path for the checkpoints
            ckpt_save_path = self.save_path / "fed_checkpoints"
            ckpt_save_path.mkdir(parents=True, exist_ok=True)

            # Extract and handle the p_client_dropout
            p_client_dropout = strategy.pop("p_client_dropout", 0.0)
            if p_client_dropout < 0.0 or p_client_dropout > 1.0:
                raise ValueError(
                    "The client dropout probability should be between 0 and 1."
                )

            # Modify the strategy to save the model
            strategy_class = set_strategy(
                strategy=strategy.pop("_target_"),
                p_client_dropout=p_client_dropout,
                model=self.c_model,
                head=self.head,
                dts_name=self.dataloaders.partitioners[
                    "train"
                ].dataset.__class__.__name__,
                save_every=log_every,
                save_path=ckpt_save_path,
                data_logger=self.data_logger,
                num_tot_rounds=num_rounds,
                class_debias=class_debias,
                class_debias_weight=class_debias_weight,
            )

            # Instantiate all the child parameters
            for key, value in strategy.items():
                try:
                    strategy[key] = instantiate(value)
                except InstantiationException:
                    strategy[key] = value

            # Define the evaluate function
            if "evaluate_fn" not in strategy:
                evaluate_fn = get_evaluate_fn(
                    model=self.c_model,
                    s_model=self.s_model,
                    head=self.head,
                    device=self.device,
                    criterion=self.criterion,
                    dataloader=self.full_test_dataloader,
                    data_logger=self.data_logger,
                    eval_every=eval_every,
                )
            else:
                evaluate_fn = strategy["evaluate_fn"]

            # Save the eval_fn as a class attribute
            # this is used to perform the eval_only test
            self.evaluate_fn = evaluate_fn

            # Filter the kwargs
            strategy_class = filter_kwargs(strategy_class)

            # Instantiate the modified strategy
            strategy = strategy_class(
                **strategy,
                initial_parameters=ndarrays_to_parameters(params),
                evaluate_fn=evaluate_fn,
                model=self.c_model,
                head=self.head,
                s_model=self.s_model,
                optimizer=self.optimizer,
                generator=self.generator,
            )

            # If required add server-side differential privacy
            if server_dp is not None:
                # Check if the dp is a DictConfig
                if not isinstance(server_dp, DictConfig):
                    raise ValueError(
                        f"DP type {type(server_dp)} not accepted. Choose one of {DictConfig}"
                    )

                # Add DP to the previously defined strategy
                strategy = instantiate(server_dp, strategy=strategy)

        # Generate the server_fn function
        server_fn = generate_server_fn(strategy=strategy, num_rounds=num_rounds)

        # Set the server
        self.server = ServerApp(server_fn=server_fn)

    def __call__(
        self, num_clients: int, log_clients_print: Optional[bool] = False
    ) -> None:
        """
        Train the model on the training set.

        Args:
            num_clients (int): The number of clients to use.
            log_clients_print (Optional[bool]): Whether to log the clients print.
                Defaults to False.

        Returns:
            None
        """

        backend_config = {"client_resources": None}
        if self.device.type == "cuda":
            backend_config = {
                "client_resources": {"num_cpus": 1, "num_gpus": 1 / 10},
                "actor": "torch",
                "init_args": {
                    "logging_level": 30,
                    "log_to_driver": log_clients_print,  # Allow each client to print to console
                },
            }

        # Set up the MLFlow parameters
        if list(self.data_logger.keys())[0] == "mlflow":
            run_name = self.data_logger["mlflow"]["run_name"]
            run_id = self.data_logger["mlflow"]["run_id"]
            tags = {
                "federated_dts": self.dts_name,
                "federated_client_model": self.c_model.__class__.__name__,
            }
            params = {
                "executed_command": self.data_logger["mlflow"]["executed_command"],
                "experiment_save_folder": self.save_path,
                **self.data_logger["mlflow"]["config_params"],
            }
        else:
            run_name = None
            run_id = None
            tags = None
            params = None

        # Train the model
        with (
            mlflow.start_run(
                run_name=run_name,
                run_id=run_id,
                log_system_metrics=False,
                tags=tags,
            )
            if list(self.data_logger.keys())[0] == "mlflow"
            else nullcontext()
        ):
            # Log training parameters on MLFlow
            # and add the run_id to the data_logger
            if (
                self.data_logger is not None
                and list(self.data_logger.keys())[0] == "mlflow"
            ):
                mlflow.log_params(params)

                self.data_logger["mlflow"]["run_id"] = mlflow.active_run().info.run_id

            # Run the simulation
            run_simulation(
                client_app=self.client,
                server_app=self.server,
                num_supernodes=num_clients,
                backend_config=backend_config,
            )

    def eval_only(self) -> None:
        """
        Perform only the evluation of both the client and server
        model skipping the federated training.

        Args:
            config (Dict[str, Scalar]): The configuration of the evaluation.
                Defaults to an empty dict.
            server_round (int): The server round to evaluate. Defaults to 0.

        Returns:
            None
        """

        # Check if the evaluate function is set
        if self.evaluate_fn is None:
            raise ValueError("The evaluate function is not set.")

        # Get the weights of the model
        models = OrderedDict({"model": self.c_model, "head": self.head})
        params = get_weights(models)

        # Call the evaluate function
        metrics = self.evaluate_fn(
            server_round=0,
            parameters=params,
            config={"force_eval": True},  # Empty config for evaluation
        )

        # Log the metrics if the data logger is set
        logger.info("Eval results:")
        for key, value in metrics[1].items():
            if "accuracy" in key.split("_") or "ece" in key.split("_"):
                value = value * 100
                suffix = "%"
            else:
                suffix = ""
            logger.info(
                "%s: %s%s",
                key.replace("_", " "),
                f"{value:.2f}" if isinstance(value, float) else value,
                suffix,
            )

    def plot_partitions(
        self,
        partitioner: Optional[str] = "train",
        custom_save_path: Optional[Union[str, Path]] = None,
        cmap: Optional[str] = None,
        bar_plot_size: Optional[Tuple[int]] = None,
        heatmap_size: Optional[Tuple[int]] = None,
    ) -> None:
        """
        Plot the parameters of the model.

        Args:
            partitioner (Optional[str]): The partitioner to use.
                Defaults to "train".
            custom_save_path (Optional[Union[str, Path]]): The path where to save the plot.
                Defaults to None. If None, the plot will be saved in the data path proided
                in the constructor.
            cmap (Optional[str]): The colormap to use for the plot.
                Defaults to None.
            bar_plot_size (Optional[Tuple[int]]): The size of the bar plot.
                Defaults to None.
            heatmap_size (Optional[Tuple[int]]): The size of the heatmap.
                Defaults to None.

        Returns:
            None
        """

        # Get the correct partitioner
        partitioner = self.dataloaders.partitioners[partitioner]
        partitioner_type = partitioner.__class__.__name__
        dataset_name = partitioner.dataset.__class__.__name__

        # Defien the save_path
        if custom_save_path is not None:
            # Check if the custom_save_path is a string
            # and convert it to a Path object
            if isinstance(custom_save_path, str):
                save_path = Path(custom_save_path)
            else:
                save_path = custom_save_path
        else:
            save_path = (
                self.save_path / f"partitions_plots/{partitioner_type}/{dataset_name}"
            )

        # Create the save_path if it does not exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Get the plot of the label distribution
        logger.info("Plotting the label distribution...")
        abs_bar_plot, _, df = plot_label_distributions(
            data=self.dataloaders.partitioners["train"],
            label_name=1,
            cmap=cmap,
            figsize=bar_plot_size,
            tight_layout=True,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=False,
            verbose_labels=True,
            legend_kwargs={"ncol": 20},
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        abs_bar_plot.savefig(save_path / "abs_bar_plot.pdf", bbox_inches="tight")
        plt.close(abs_bar_plot)
        logger.info("Absolute bar plot created.")

        perc_bar_plot, _, _ = plot_label_distributions(
            data=self.dataloaders.partitioners["train"],
            label_name=1,
            cmap=cmap,
            figsize=bar_plot_size,
            tight_layout=True,
            plot_type="bar",
            size_unit="percent",
            partition_id_axis="x",
            legend=False,
            verbose_labels=True,
            legend_kwargs={"ncol": 20},
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        perc_bar_plot.savefig(save_path / "perc_bar_plot.pdf", bbox_inches="tight")
        plt.close(perc_bar_plot)
        logger.info("Percentage bar plot created.")

        heatmap, _, _ = plot_label_distributions(
            data=copy(df),
            label_name=1,
            figsize=heatmap_size,
            tight_layout=True,
            plot_type="heatmap",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            cmap="inferno",
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        heatmap.savefig(save_path / "heatmap.pdf", bbox_inches="tight")
        plt.close(heatmap)
        logger.info("Heatmap created.")
        logger.info("Label distribution plots created.")  #
