"""
This module contains the class to perform the pretraining step.

Classes:
    Pretraining: Class to perform the pretraining step.

Functions:
    None

Constants:
    None

Exceptions:
    ValueError: Raised when the server model is not accepted.
    ValueError: Raised when the client model is not accepted.
    ValueError: Raised when the head model is not accepted.
    ValueError: Raised when the criterion and the KD loss are not provided.
    ValueError: Raised when the data logger is not accepted.
    ValueError: Raised when the checkpoint module is not accepted.
    AssertionError: Raised when the dataloaders are instantiated with an illegal id.

Authors: Matteo Caligiuri
         Francesco Barbato
"""

import os
from typing import Optional, Union, Dict, Literal
from pathlib import Path
from contextlib import nullcontext

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.transforms.v2 import Compose
from torch.utils.tensorboard import SummaryWriter
import mlflow
import numpy as np

from modules.trainers.pretraining_task import run_epoch
from modules.common.auxiliary_fn import TimeEstimator, filtered_instantiate
from modules.common.logger import Logger, TqdmLogger
from modules.common.constants import ALLOWED_MODULE2TRAIN, ACCEPTED_DATA_LOGGER
from modules.common.training_utils import (
    set_optimizer,
    set_train_mode,
    set_eval_mode,
)
from modules.common.data_types import TorchModule
from modules.datasets.data_handler import load_dts

# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


# Export necessary env variable
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Define wahat to export
__all__ = ["Pretraining"]


class Pretraining:
    """
    Class to perform the pretraining step.

    Args:
        server (Union[DictConfig, torch.nn.Module]): The server model.
        client (Union[DictConfig, torch.nn.Module]): The client model.
        head (Union[DictConfig, torch.nn.Module]): The head of the model.
        n_classes (int): The number of classes.
        device (Optional[Union[str, torch.device]], optional): The device to use for training.
            Defaults to "cpu".
        data_folder (Optional[Union[Path, str]], optional): The path to the data folder.
        checkpoint (Optional[Union[Path, str]], optional): The path to the pretraining checkpoint.
            Defaults to None.
        eval (bool, optional): If True, the model is in evaluation mode. Defaults to False.
        data_logger (Optional[Dict[str, Union[SummaryWriter, Dict[str, str], None]]], optional):
            The data logger. Defaults to None.

    Returns:
        None
    """

    def __init__(
        self,
        server: Union[DictConfig, torch.nn.Module],
        client: Union[DictConfig, torch.nn.Module],
        head: Union[DictConfig, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = "cpu",
        data_folder: Optional[Union[Path, str]] = Path("data"),
        checkpoint: Optional[Union[Path, str]] = None,
        eval: bool = False,  # pylint: disable=redefined-builtin
        data_logger: Optional[
            Dict[str, Union[SummaryWriter, Dict[str, str], None]]
        ] = None,
        dataloader_workers: Optional[int] = 4,
    ) -> None:
        # Parse the device to torch.device
        if isinstance(device, str):
            device = torch.device(device)

        # Check the data folder and create it if it does not exist
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)

        data_folder.mkdir(parents=True, exist_ok=True)

        self.data_folder = data_folder

        # Define the server and client models
        if isinstance(server, torch.nn.Module):
            self.s_model = server
        elif isinstance(server, DictConfig):
            self.s_model = instantiate(
                server,
                device=device,
                data_folder=data_folder,
            )
        else:
            raise ValueError(f"Server model {type(server)} not accepted.")
        if isinstance(client, torch.nn.Module):
            self.c_model = client
        elif isinstance(client, DictConfig):
            self.c_model = instantiate(
                client,
                out_feats_size=self.s_model.get_feats_shape(),
            ).to(device)
        else:
            raise ValueError(f"Client model {type(client)} not accepted.")

        # Define the head model module
        self.head_module = head

        # Initialize the head dict
        self.heads = {}

        # Define the device
        self.device = device

        # Set the evaluation flag
        self.eval = eval

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

        # Initialize the dataloaders
        self.dataloaders = None
        self.dataloaders_a1 = None
        self.dataloaders_a2 = None

        # Initialize the label mapping
        self.label_mapping = None
        self.label_mapping_a1 = None
        self.label_mapping_a2 = None

        # Initialize the optimizer and loss function
        self.criterion = None
        self.kd_loss = None
        self.optimizer = None
        self.scheduler = None

        # Initialize the module2train var
        self.module2train = None

        # Initialize the loss weight and the KD loss weight
        self.loss_weight = 0.5
        self.kd_loss_weight = 0.5
        self.max_epoch_iters = None

        # Check if a pretraining checkpoint is provided
        if checkpoint is not None:
            # Load the pretraining checkpoint
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: DictConfig) -> None:
        """
        Load the pretraining checkpoint.

        Args:
            checkpoint (DictConfig): The path to the checkpoint.

        Returns:
            None
        """

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

        # Load the model
        if checkpoint_module == "all":
            self.c_model.load_state_dict(checkpoint_data["model"], strict=False)
            for k, v in self.heads.items():
                v.load_state_dict(checkpoint["heads"][k])
        elif checkpoint_module == "main":
            self.c_model.load_state_dict(checkpoint_data["model"], strict=False)
        elif checkpoint_module == "head":
            for k, v in self.heads.items():
                v.load_state_dict(checkpoint["heads"][k])

        # Load the scheduler
        if checkpoint_data["scheduler"] is not None:
            self.scheduler = {
                "data": checkpoint_data["scheduler"],
                "epoch": checkpoint_data["epoch"],
            }
        else:
            self.scheduler = None

        # Load the optimizer
        # self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save_checkpoint(
        self, path: Path, epoch: int, last: Optional[bool] = False
    ) -> None:
        """
        Save the pretraining checkpoint.

        Args:
            path (Path): The path to save the checkpoint.
            epoch (int): The current epoch.
            last (Optional[bool], optional): If True, the checkpoint is the last one.
                Defaults to False.

        Returns:
            None
        """

        # Define the checkpoint name
        c_model_name = self.c_model.__class__.__name__.lower()
        index = "last" if last else f"{epoch+1:03d}"
        pretraining_ckpt_name = f"{index}_{c_model_name}_pretraining_checkpoint.pth"

        # Save the checkpoint
        torch.save(
            {
                "model": self.c_model.state_dict(),
                "heads": {k: v.state_dict() for k, v in self.heads.items()},
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "epoch": epoch,
                "criterion": self.criterion,
                "kd_loss": self.kd_loss,
            },
            path / pretraining_ckpt_name,
        )

        # If MLFlow is used save the model there
        if list(self.data_logger.keys())[0] == "mlflow" and last:
            mlflow.pytorch.log_model(
                self.c_model,
                pretraining_ckpt_name.split(".", maxsplit=1)[0],
                signature=False,
            )

    def set_data(
        self,
        dataset_info: DictConfig,
        batch_size: int,
        train_transforms: Optional[Union[Compose, DictConfig]] = None,
        val_transforms: Optional[Union[Compose, DictConfig]] = None,
        generator: Optional[torch.Generator] = None,
        is_additional: Optional[Literal[0, 1, 2]] = 0,
    ) -> None:
        """
        Set the pretraining data.

        Args:
            dataset_info (DictConfig): The info of the dataset to load.
            batch_size (int): The batch size.
            global_cfg (DictConfig): Global config dictionary,
                needed to retrieve the embedding model.
            transforms (Optional[Union[Compose, DictConfig]], optional): The transforms to apply.
                Defaults to None.
            generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
            is_additional (Optional[Literal[0, 1, 2]], optional): Wether the dataset main
                or additional.
                Defaults to 0 == main dataset, 1: additional dataset 1, 2: additional dataset 2.
        Returns:
            None
        """
        assert is_additional in [
            0,
            1,
            2,
        ], f"Illegal Value {is_additional} for is_additional, must be [0,1,2]"

        # Check if the transforms are provided
        # and parse them from DictConfig to Compose
        # if necessary
        if train_transforms is not None and isinstance(train_transforms, DictConfig):
            train_transforms = instantiate(train_transforms)
        if val_transforms is not None and isinstance(val_transforms, DictConfig):
            val_transforms = instantiate(val_transforms)

        transforms = {"train": train_transforms, "val": val_transforms}

        # Load the dataloaders of the pretraining dataset and the label mapping
        logger.info(
            "Loading pretraining dataset dataset id: %d (%s) ...",
            is_additional,
            dataset_info.name,
        )
        dataloaders, label_mapping = load_dts(
            dts_info=dataset_info,
            batch_size=batch_size,
            cache_dir=self.data_folder,
            teacher_name=self.s_model.__class__.__name__.lower(),
            transforms=transforms,
            generator=generator,
            num_workers=self.dataloader_workers,
        )

        # Set the dataloaders and label mapping
        # for default (is_additional == 0) or
        # additional datasets
        if is_additional == 0:
            self.dataloaders = dataloaders
            self.label_mapping = label_mapping

        if is_additional == 1:
            self.dataloaders_a1 = dataloaders
            self.label_mapping_a1 = label_mapping

        if is_additional == 2:
            self.dataloaders_a2 = dataloaders
            self.label_mapping_a2 = label_mapping

    def evaluate(
        self,
        current_epoch: Optional[int] = 0,
        log_every: Optional[int] = 10,
        max_iters: Optional[int] = None,
    ) -> None:
        """
        Evaluate the model.

        Args:
            current_epoch (Optional[int], optional): The current epoch. Defaults to None.
            log_every (Optional[int], optional): The number of batches to wait before logging.

        Returns:
            None
        """

        # Initialize max_iters if it's not provided
        if max_iters is None:
            max_iters = len(self.dataloaders["val"])

        # Initialize the time estimator
        e_time = TimeEstimator(total=max_iters)

        # Set the models in evaluation mode
        set_eval_mode(
            model=self.c_model,
            head=self.heads,
            s_model=self.s_model,
        )

        # Evaluate the model
        logger.info("Starting evaluation...")
        e_time.start()

        # intialize dataloaders list and run eval epoch
        dls = [self.dataloaders["val"]]
        if self.dataloaders_a1 is not None:
            dls.append(self.dataloaders_a1["val"])
        if self.dataloaders_a2 is not None:
            dls.append(self.dataloaders_a2["val"])
        out = run_epoch(
            c_model=self.c_model,
            s_model=self.s_model,
            heads=self.heads,
            epoch=current_epoch,
            dataloaders=dls,
            time_estimator=e_time,
            device=self.device,
            data_logger=self.data_logger,
            log_every=log_every,
            eval=True,
            max_iters=max_iters,
        )

        # Log the final metrics
        logger.info("")
        logger.info(
            "Evaluation completed in %s | Accuracy-1: [server=%.2f%%, client=%.2f%%] |"
            + "Accuracy-5: [server=%.2f%%, client=%.2f%%]",
            e_time.get_total(h_format=True),
            np.mean(out["s_accuracy_1"]) * 100,
            np.mean(out["c_accuracy_1"]) * 100,
            np.mean(out["s_accuracy_5"]) * 100,
            np.mean(out["c_accuracy_5"]) * 100,
        )

        # Set the models in training mode
        set_train_mode(
            model=self.c_model,
            head=self.heads,
            s_model=self.s_model,
            module2train=self.module2train,
        )

    def init_classifiers(self) -> None:
        """
        Initialize classifiers for the main, aux 1, and aux 2 datasets.

        This method initializes SimpleClassifier instances
        for each dataset specified in the `dataloaders` dictionary.
        The classifier is initialized with the feature shape of the
        model features and the number of classes of each dataset.

        Returns:
            None
        """

        if isinstance(self.head_module, torch.nn.Module):
            classisfier_cls = self.head_module
        elif isinstance(self.head_module, DictConfig):
            classisfier_cls = instantiate(self.head_module, _partial_=True)
        else:
            raise ValueError(f"Head model {type(self.head_module)} not accepted.")

        self.heads[self.dataloaders["train"].dataset.name] = classisfier_cls(
            features=self.s_model.get_feats_shape(),
            num_classes=self.dataloaders["train"].dataset.num_classes,
        )
        if self.dataloaders_a1 is not None:
            self.heads[self.dataloaders_a1["train"].dataset.name] = classisfier_cls(
                features=self.s_model.get_feats_shape(),
                num_classes=self.dataloaders_a1["train"].dataset.num_classes,
            )
        if self.dataloaders_a2 is not None:
            self.heads[self.dataloaders_a2["train"].dataset.name] = classisfier_cls(
                features=self.s_model.get_feats_shape(),
                num_classes=self.dataloaders_a2["train"].dataset.num_classes,
            )

        # move classifiers to the compute device
        for k in self.heads:
            self.heads[k] = self.heads[k].to(self.device)

    def __call__(
        self,
        n_epochs: int,
        optim: DictConfig,
        kd_loss: DictConfig,
        save_path: Path,
        log_every: Optional[int] = 10,
        evaluate_every: Optional[int] = None,
        loss: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        loss_weight: Optional[float] = 0.5,
        kd_loss_weight: Optional[float] = 0.5,
        early_stopping: Optional[bool] = True,
        max_epoch_iters: Optional[int] = None,
        module2train: Optional[str] = "all",
    ) -> Optional[Dict[str, TorchModule]]:
        """
        Train the model.

        Args:
            n_epochs (int): The number of epochs.
            optim (DictConfig): The optimizer.
            kd_loss (DictConfig): The knowledge distillation loss function.
            save_path (Path): The path to save the model.
            log_every (Optional[int], optional): The number of batches to wait before logging.
            evaluate_every (Optional[int], optional): The number of epochs to wait before
                evaluating the model. Defaults to None. If None, the model is evaluated at the end
                of the training.
            loss (Optional[DictConfig], optional): The loss function. Defaults to None.
            scheduler (Optional[DictConfig], optional): The scheduler. Defaults to None.
            loss_weight (Optional[float], optional): The weight of the KD loss. Defaults to 0.5.
            kd_loss_weight (Optional[float], optional): The weight of the KD loss. Defaults to 0.5.
            early_stopping (bool, optional): If True, perform early stopping. Defaults to True.
            max_epoch_iters (Optional[int], optional): The maximum number of iterations per epoch.
                Defaults to None.
            module2train (Optional[str], optional): The module to train. Defaults to "all".
        Returns:
            Optional[Dict[str, TorchModule]]: The trained model.
                A dict with the keys "model" and "heads".
                model is the trained client model, heads is a dict with the trained classifiers.
                If it is called with eval=True, it returns None.
        """

        self.max_epoch_iters = max_epoch_iters

        # Define the optimizer and loss function
        self.criterion = instantiate(loss) if loss is not None else None
        self.kd_loss = instantiate(kd_loss)
        self.optimizer, self.module2train = set_optimizer(
            model=self.c_model,
            head=self.heads,
            optimizer=optim,
            module2train=module2train,
        )

        # Set the right module to train mode
        set_train_mode(
            model=self.c_model,
            head=self.heads,
            s_model=self.s_model,
            module2train=self.module2train,
        )

        # Define or load the scheduler
        # If the scheduler is already defined, load the state
        if self.scheduler is not None:
            data = self.scheduler["data"]
            self.scheduler = filtered_instantiate(
                scheduler,
                optimizer=self.optimizer,
                T_max=n_epochs * len(self.dataloaders["train"]),
                last_epoch=-1,
            ).load_state_dict(data)
            del data
        # If the scheduler is not defined, create a new one
        elif scheduler is not None:
            self.scheduler = filtered_instantiate(
                scheduler,
                optimizer=self.optimizer,
                last_epoch=-1,
                T_max=n_epochs * len(self.dataloaders["train"]),
            )

        # Set the loss weight and the KD loss weight
        if self.criterion and self.kd_loss:
            self.loss_weight = loss_weight
        elif self.criterion and not self.kd_loss:
            self.loss_weight = 0.0
        else:
            self.loss_weight = 1.0
        self.kd_loss_weight = kd_loss_weight

        # Train the model
        # Use the server model as a teacher (no training just inference)
        # And perform KD to the client model

        # Create the labels embeddings for CLIP
        if hasattr(self.s_model, "create_labels_embeddings"):
            self.s_model.create_labels_embeddings(
                labels=self.label_mapping,
                normalize=True,
                prepend_phrase="A low contrast photo of a ",
            )

        if not self.eval:
            # Check that at least one between the criterion and the KD loss is not None
            if self.criterion is None and self.kd_loss is None:
                raise ValueError(
                    "At least one between the criterion and the KD loss must be provided"
                )

            # Initialize the time estimator
            e_time = TimeEstimator(total=n_epochs)

            # Initialize the accuracy
            s_accuracy_1 = np.zeros(n_epochs)
            s_accuracy_5 = np.zeros(n_epochs)
            c_accuracy_1 = np.zeros(n_epochs)
            c_accuracy_5 = np.zeros(n_epochs)

            # Init last epoch
            last_epoch = 0

            # Define the name of the run with the relatives tags and params if MLFlow is used
            if list(self.data_logger.keys())[0] == "mlflow":
                run_name = self.data_logger["mlflow"]["run_name"]
                run_id = self.data_logger["mlflow"]["run_id"]
                tags = {
                    "pretraining_dts": self.dataloaders["train"].dataset.name,
                    "pretraining_model": self.c_model.__class__.__name__,
                }
                params = {
                    "executed_command": self.data_logger["mlflow"]["executed_command"],
                    "experiment_save_folder": save_path,
                    **self.data_logger["mlflow"]["config_params"],
                }
            else:
                run_name = None
                run_id = None
                tags = None
                params = None

            # Train the model
            logger.info("Starting pre-training...")
            e_time.start()

            with (
                mlflow.start_run(
                    run_id=run_id,
                    run_name=run_name,
                    log_system_metrics=True,
                    tags=tags,
                )
                if list(self.data_logger.keys())[0] == "mlflow"
                else nullcontext()
            ):
                # Log training parameters on MLFlow
                # and the run id on the logger
                if (
                    self.data_logger is not None
                    and list(self.data_logger.keys())[0] == "mlflow"
                ):
                    mlflow.log_params(params)
                    self.data_logger["mlflow"][
                        "run_id"
                    ] = mlflow.active_run().info.run_id

                for epoch in range(n_epochs):
                    # intialize dataloaders list and run train epoch
                    dls = [self.dataloaders["train"]]
                    if self.dataloaders_a1 is not None:
                        dls.append(self.dataloaders_a1["train"])
                    if self.dataloaders_a2 is not None:
                        dls.append(self.dataloaders_a2["train"])

                    out = run_epoch(
                        c_model=self.c_model,
                        s_model=self.s_model,
                        heads=self.heads,
                        epoch=epoch,
                        dataloaders=dls,
                        time_estimator=e_time,
                        device=self.device,
                        criterion=self.criterion,
                        kd_loss=self.kd_loss,
                        loss_weight=self.loss_weight,
                        kd_loss_weight=self.kd_loss_weight,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        data_logger=self.data_logger,
                        tot_epoch=n_epochs,
                        log_every=log_every,
                        eval=False,
                        max_iters=self.max_epoch_iters,
                    )

                    # Save the accuracy
                    s_accuracy_1[epoch] = out["s_accuracy_1"]
                    s_accuracy_5[epoch] = out["s_accuracy_5"]
                    c_accuracy_1[epoch] = out["c_accuracy_1"]
                    c_accuracy_5[epoch] = out["c_accuracy_5"]

                    # Update the last epoch
                    last_epoch = epoch

                    # Save intermediate checkpoints
                    if (epoch + 1) % 10 == 0:
                        self.save_checkpoint(path=save_path, epoch=epoch)

                    # Evaluate the model
                    if evaluate_every is not None and (epoch + 1) % evaluate_every == 0:
                        self.evaluate(
                            current_epoch=epoch + 1 // evaluate_every,
                            log_every=log_every,
                            max_iters=self.max_epoch_iters,
                        )

                    # Early stopping
                    # Stop the training if the accuracy is not improving from the last 5 epochs
                    if (
                        np.all(
                            np.abs(np.diff(c_accuracy_1[epoch - 5 : epoch + 1])) <= 5e-4
                        )
                        and epoch > 5
                        and early_stopping
                    ):
                        logger.info(
                            "Early stopping at epoch %d due to lack of improvements"
                            + " in the last 5 epochs",
                            epoch,
                        )
                        break

                # Save the model
                self.save_checkpoint(path=save_path, epoch=last_epoch, last=True)

                # Log the final metrics
                logger.info("")
                logger.info(
                    "Pre-training completed in %s | Accuracy-1: [server=%.2f%%, client=%.2f%%] | "
                    + "Accuracy-5: [server=%.2f%%, client=%.2f%%]",
                    e_time.get_total(h_format=True),
                    s_accuracy_1[-1] * 100,
                    c_accuracy_1[-1] * 100,
                    s_accuracy_5[-1] * 100,
                    c_accuracy_5[-1] * 100,
                )

                # Evaluate the model if has not been evaluated during the training
                if evaluate_every is None or (last_epoch + 1) % evaluate_every != 0:
                    self.evaluate(current_epoch=last_epoch, log_every=log_every)

        else:
            self.evaluate(current_epoch=0, log_every=log_every)

        # Return the trained models
        return {"model": self.c_model, "heads": self.heads}
