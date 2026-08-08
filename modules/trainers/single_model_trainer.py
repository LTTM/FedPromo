"""
This module include a standard set of function to train and/or evaluate a single model.

Classes:
    Trainer: A class to train a model.

Functions:
    None

Constants:
    ALLOWED_PARAM2TRAIN: The allowed parameters to train the model.

Exceptions:
    None

Author: Matteo Caligiuri
"""

from typing import Optional, Dict, Union
from pathlib import Path
from datetime import datetime as date
from contextlib import nullcontext, redirect_stderr, redirect_stdout

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Compose
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from modules.common import (
    Logger,
    TimeEstimator,
    ACCEPTED_DATA_LOGGER,
    log2logger,
    TqdmLogger,
    normalize_tensor,
    filtered_instantiate,
    set_optimizer,
    set_train_mode,
    set_eval_mode,
)
from modules.datasets import load_dts


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["Trainer"]


class Trainer:
    """
    Class to train a model.

    Args:
        model (nn.Module): The model to train.
        head (nn.Module): The head of the model.
        n_classes (int): The number of classes.
        device (torch.device): The device to use.
        loss (DictConfig): The loss function to use.
        optimizer (DictConfig): The optimizer to use.
        module2train (Optional[str]): The module to train. Defaults to "all".
        proto_train (Optional[bool]): Train using the prototypes.
            Defaults to False.
        data_folder (Optional[Union[Path, str]]): The path to the data folder.
            Defaults to Path("data").
        save_folder (Optional[Union[Path, str]]): The path to the save folder.
            Defaults to Path("save").
        checkpoint (Optional[Union[Path, str]]): The path to the checkpoint. Defaults to None.
        eval (Optional[bool]): Perform only evaluation of the model. Defaults to False.
        data_logger (Optional[Dict[str, Union[SummaryWriter, str, None]]]): The data logger.
            Defaults to None.
        log_every (Optional[int]): The number of batches to log. Defaults to 10.
        evaluate_every (Optional[int]): The number of epochs to evaluate the model.
            Defaults to None.
        early_stopping (bool): Perform early stopping. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        n_classes: int,
        device: torch.device,
        loss: DictConfig,
        optimizer: DictConfig,
        module2train: Optional[str] = "all",
        proto_train: Optional[bool] = False,
        data_folder: Optional[Union[Path, str]] = Path("data"),
        save_folder: Optional[Union[Path, str]] = Path("save"),
        checkpoint: Optional[Union[Path, str]] = None,
        eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
        data_logger: Optional[Dict[str, Union[SummaryWriter, str, None]]] = None,
        log_every: Optional[int] = 10,
        evaluate_every: Optional[int] = None,
        early_stopping: bool = False,
    ) -> None:
        # Check the data folder and create it if it does not exist
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)

        data_folder.mkdir(parents=True, exist_ok=True)

        self.data_folder = data_folder

        # Define the prototypes variable and folder
        self.prototypes = None
        self.prototypes_folder = data_folder / "prototypes"
        self.prototypes_folder.mkdir(parents=True, exist_ok=True)

        # Define the model
        # Move it to the correct device
        if not hasattr(model, "device"):
            model.to(device)
        self.model = model
        self.head = head.to(device)

        # Define the device
        self.device = device

        # Define the number of classes
        self.n_classes = n_classes

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

        # Define oher logging variables
        self.log_every = log_every
        self.evaluate_every = evaluate_every

        # Initialize the dataloaders
        self.dataloaders = None

        # Initialize the label mapping
        self.label_mapping = None

        # Define the optimizer and loss function
        self.criterion = instantiate(loss)
        self.optimizer, self.module2train = set_optimizer(
            model=model, head=head, optimizer=optimizer, module2train=module2train
        )

        # Define the scheduler
        self.scheduler = None

        # Create the labels embeddings for CLIP
        if hasattr(self.model, "create_labels_embeddings"):
            self.model.create_labels_embeddings(
                labels=self.label_mapping,
                normalize=True,
                prepend_phrase="A low contrast photo of a ",
            )

        # Set the variable that defines if the model is training with prototypes or not
        self.proto_train = proto_train

        # Define the save folder
        if isinstance(save_folder, str):
            save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        self.save_folder = save_folder

        # Define the early stopping flag
        self.early_stopping = early_stopping

        # Check if a pretraining checkpoint is provided
        if checkpoint is not None:
            # Cast the path to Path if it is a string
            if isinstance(checkpoint, str):
                checkpoint = Path(checkpoint)

            # Load the pretraining checkpoint
            self._load_checkpoint(**checkpoint)

    def set_data(
        self,
        dataset_info: DictConfig,
        batch_size: int,
        transforms: Optional[Dict[str, Compose]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Set the pretraining data.

        Args:
            dataset_info (DictConfig): The info of the dataset to load.
            batch_size (int): The batch size.
            transforms (Optional[Dict[str, Compose]], optional): The transformations to apply.
                Defaults to None. If None, the default transformation is applied (ToTensor(image)).
            generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
            splits (Optional[Union[str, list]], optional): The splits to load.
                Defaults load all tha available splits.
        Returns:
            None
        """

        # Load the dataloaders of the pretraining dataset and the label mapping
        logger.info("Loading pretraining dataset...")
        dataloaders, label_mapping = load_dts(
            dts_info=dataset_info,
            batch_size=batch_size,
            transforms=transforms,
            generator=generator,
        )

        # Set the dataloaders
        self.dataloaders = dataloaders

        # Set the label mapping
        self.label_mapping = label_mapping

    def _load_checkpoint(
        self,
        path: Union[str, Path],
        module2load: Optional[str] = "all",
        ignore_scheduler: Optional[bool] = True,
    ) -> None:
        """
        Load a checkpoint.

        Args:
            path (Union[str, Path]]): The path to the checkpoint.
            module2load (Optional[str]): The module to load.
                Defaults to all.
            ignore_scheduler (Optional[bool]): If True,
                the scheduler is not loaded.

        Returns:
            None
        """

        # If necessary cast the path to Path
        if isinstance(path, str):
            path = Path(path)

        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False)

        # Load the model
        if module2load == "all":
            self.model.load_state_dict(checkpoint["model"])
            self.head.load_state_dict(checkpoint["head"])
        elif module2load == "head":
            self.head.load_state_dict(checkpoint["head"])
        elif module2load == "main":
            self.model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError(f"Module {module2load} not accepted")

        # Load the scheduler
        if (
            "scheduler" in checkpoint
            and checkpoint["scheduler"] is not None
            and not ignore_scheduler
        ):
            self.scheduler = {
                "data": checkpoint["scheduler"],
                "epoch": checkpoint["epoch"],
            }
        else:
            self.scheduler = None

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        module: Optional[str] = None,
        last: Optional[bool] = False,
    ) -> None:
        """
        Save the pretraining checkpoint.

        Args:
            path (Path): The path to save the checkpoint.
            epoch (int): The current epoch.
            module (Optional[str], optional): The module to save. Defaults to None.
                If None, the whole model is saved.
            last (Optional[bool], optional): If True, the checkpoint is the last one.
                Defaults to False.

        Returns:
            None
        """

        # Define the checkpoint name
        model_name = self.model.__class__.__name__.lower()
        index = "last" if last else f"{epoch+1:03d}"
        pretraining_ckpt_name = f"{index}_{model_name}_pretraining_checkpoint.pth"

        # Save the checkpoint
        torch.save(
            {
                "model": (
                    self.model.state_dict()
                    if module is None
                    else getattr(self.model, module).state_dict()
                ),
                "head": self.head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "epoch": epoch,
                "criterion": self.criterion,
            },
            path / pretraining_ckpt_name,
        )

        # If MLFlow is used save the model there
        if list(self.data_logger.keys())[0] == "mlflow" and last:
            mlflow.pytorch.log_model(
                self.model, pretraining_ckpt_name.split(".")[0], signature=False
            )

    def _compute_loss_and_predictions(
        self, out: Dict[str, torch.Tensor], label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get the class predictions and the loss of the model.
        It return the correct class prediction for the model
        in both the cases the prediction is the argmax of the logits
        or the argmin/argmax of the distance between the logits and the prototypes.
        It also return the loss of the model if the label is provided.

        Args:
            out (Dict[str, torch.Tensor]): The output of the model.
            label (Optional[torch.Tensor]): The labels.
                Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The class predictions
                and the loss of the model.
                ["pred"]: The class predictions.
                ["loss"]: The loss of the model.
        """

        # Initialize the return dictionary
        data = {}

        # Get the correct output
        if self.prototypes is not None:
            # Get the output features
            try:
                feats = out["feats"]
            except TypeError:
                feats = out[1]

            # Prepare the inpud data
            feats = feats.unsqueeze(1).repeat(1, self.n_classes, 1)
            prototypes = self.prototypes.unsqueeze(0).repeat(feats.size(0), 1, 1)

            # Compute the distances
            dist = torch.mean(self.criterion(feats, prototypes), dim=2)

            # Get the class predictions
            if not isinstance(self.criterion, torch.nn.CosineSimilarity):
                data["top_k"] = torch.topk(dist, k=5, dim=1)[1]
            else:
                data["top_k"] = torch.topk(-dist, k=5, dim=1)[1]

            # Compute the loss
            if label is not None:
                data["loss"] = torch.mean(self.criterion(feats, prototypes))
        else:
            # Get the output logits
            try:
                logits = out["logits"]
            except TypeError:
                logits = out[0]

            # Get the class predictions
            data["top_k"] = torch.topk(logits, k=5, dim=1)[1]

            # Compute the loss
            if label is not None:
                data["loss"] = self.criterion(logits, label)

        return data

    def _run_batch(
        self,
        batch_index: int,
        tot_batches: int,
        current_epoch: int,
        batch: Dict[str, torch.Tensor],
        time_estimator: TimeEstimator,
        log_every: Optional[int] = 5,
        eval: bool = False,  # pylint: disable=redefined-builtin
    ) -> Dict[str, torch.Tensor]:
        """
        Run a batch of data through the model.

        Args:
            batch_index (int): The index of the batch.
            tot_batches (int): The total number of batches.
            current_epoch (int): The current epoch.
            batch (Dict[str, torch.Tensor]): The batch of data.
            time_estimator (TimeEstimator): The time estimator.
            log_every (Optional[int], optional): The number of batches to wait before logging.
            eval (bool, optional): If True, the model is in evaluation mode. Defaults to False.
        Returns:
            Dict[str, torch.Tensor]: The output of the model.
        """

        # Move the data to device
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)

        # Compute the model prediction
        with torch.set_grad_enabled(not self.eval):
            out = {"feats": None, "logits": None}
            out["feats"] = self.model(image)
            out["logits"] = self.head(normalize_tensor(out["feats"]))

        # Get the class predictions and also the loss (only if not in evaluation mode)
        if not eval:
            data = self._compute_loss_and_predictions(out, label)
            top_k = data["top_k"]
            loss = data["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            top_k = self._compute_loss_and_predictions(out)["top_k"]
            loss = None

        # Compute the batch accuracy
        accuracy_1 = torch.mean(1.0 * (top_k[:, 0] == label)).item()

        # Update the batch time log
        time_estimator.update()

        # Log the partial results and evaluation status
        if (batch_index + 1) % log_every == 0:
            logger.info(
                "[Batch %03d/%03d][Time for batch: %s | ETA %s] "
                + "Batch accuracy-1: %.2f%% | lr: %.6f",
                batch_index + 1,
                tot_batches,
                time_estimator.get_last(h_format=True),
                time_estimator.estimate(h_format=True),
                accuracy_1 * 100,
                self.optimizer.param_groups[0]["lr"],
            )

            # Log the batch metrics on tensorboard or MLFlow
            if self.data_logger is not None and not eval:
                data = {}
                data["Train_batch/loss"] = loss.item()
                if self.scheduler is not None:
                    data["Train_batch/Train lr"] = self.optimizer.param_groups[0]["lr"]
                log2logger(
                    logger=self.data_logger,
                    epoch=(batch_index + 1) + (current_epoch * tot_batches),
                    data=data,
                )

        out = {
            "top_k": top_k,
            "label": label,
            "loss": loss,
        }

        return out

    def _run_epoch(
        self,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        time_estimator: TimeEstimator,
        tot_epoch: Optional[int] = 0,
        log_every: Optional[int] = 5,
        eval: bool = False,  # pylint: disable=redefined-builtin
    ) -> Dict[str, torch.Tensor]:
        """
        Run an epoch of the model.

        Args:
            epoch (int): The current epoch.
            dataloader (torch.utils.data.DataLoader): The dataloader.
            time_estimator (TimeEstimator): The time estimator.
            tot_epoch (int): The total number of epochs.
            log_every (Optional[int], optional): The number of batches to wait before logging.
            eval (bool, optional): If True, the model is in evaluation mode. Defaults to False.
        Returns:
            Dict[str, torch.Tensor]: The output metrics.
        """

        # Log the epoch
        if not eval:
            logger.info(
                "%s Epoch %03d/%03d %s", "-" * 44, epoch + 1, tot_epoch, "-" * 44
            )
        else:
            logger.info("%s Val Epoch %03d %s", "-" * 44, epoch + 1, "-" * 44)

        # Initialize the predictions and labels
        top_k = None
        labels = None

        # Initialize the losses
        losses = []

        # Define the batch time estiamtor
        b_time = TimeEstimator(total=len(dataloader))

        b_time.start()
        for i, batch in enumerate(dataloader):
            b_out = self._run_batch(
                batch_index=i,
                tot_batches=len(dataloader),
                current_epoch=epoch,
                batch=batch,
                log_every=log_every,
                time_estimator=b_time,
                eval=eval,
            )

            top_k = (
                b_out["top_k"]
                if top_k is None
                else torch.cat((top_k, b_out["top_k"]), dim=0)
            )
            labels = (
                b_out["label"]
                if labels is None
                else torch.cat((labels, b_out["label"]), dim=0)
            )
            if not eval:
                losses.append(b_out["loss"].item())

        # Compute the aggregate epoch accuracy
        accuracy_1 = torch.mean(1.0 * (top_k[:, 0] == labels)).item()
        accuracy_5 = torch.mean(
            1.0 * torch.any(top_k == labels.unsqueeze(1), dim=1)
        ).item()

        # Update the epoch time log
        time_estimator.update()

        # Log the epoch metrics
        logger.info("")
        logger.info(
            "[EPOCH %03d SUMMARY][Time for epoch: %s | ETA: %s] Accuracy-1: %.2f%% | Accuracy-5: %.2f%%",  # pylint: disable=line-too-long
            epoch + 1,
            time_estimator.get_last(h_format=True),
            time_estimator.estimate(h_format=True),
            accuracy_1 * 100,
            accuracy_5 * 100,
        )
        logger.info("-" * 103)

        # Log the losses and KD loss on tensorboard or MLFlow
        if self.data_logger is not None and not eval:
            data = {}
            data["Train/loss"] = f"{np.mean(losses):4f}"
            data["Train/accuracy-1"] = accuracy_1 * 100
            data["Train/accuracy-5"] = accuracy_5 * 100

            log2logger(
                logger=self.data_logger,
                epoch=(epoch + 1),
                data=data,
            )
        elif self.data_logger is not None and eval:
            data = {}
            data["Validation/accuracy-1"] = accuracy_1 * 100
            data["Validation/accuracy-5"] = accuracy_5 * 100

            log2logger(
                logger=self.data_logger,
                epoch=(epoch + 1),
                data=data,
            )

        out = {"accuracy_1": accuracy_1, "accuracy_5": accuracy_5}

        return out

    def evaluate(
        self,
        current_epoch: Optional[int] = 0,
        log_every: Optional[int] = 10,
    ) -> None:
        """
        Evaluate the model.

        Args:
            current_epoch (Optional[int], optional): The current epoch. Defaults to None.
            log_every (Optional[int], optional): The number of batches to wait before logging.

        Returns:
            None
        """

        # Initialize the time estimator
        e_time = TimeEstimator(total=len(self.dataloaders["val"]))

        # Set the models in evaluation mode
        set_eval_mode(model=self.model, head=self.head)

        # Evaluate the model
        logger.info("Starting evaluation...")
        e_time.start()
        out = self._run_epoch(
            epoch=current_epoch,
            dataloader=self.dataloaders["val"],
            log_every=log_every,
            time_estimator=e_time,
            eval=True,
        )

        # Set the models in training mode
        self.model.train()
        self.head.train()

        # Log the final metrics
        logger.info("")
        logger.info(
            "Evaluation completed in %s | Accuracy-1: %.2f%% | Accuracy-5: %.2f%%",
            e_time.get_total(h_format=True),
            out["accuracy_1"] * 100,
            out["accuracy_5"] * 100,
        )

        # Set the models in training mode
        set_train_mode(model=self.model, head=self.head, module2train=self.module2train)

    def __call__(
        self,
        n_epochs: int,
        scheduler: Optional[DictConfig] = None,
        module_to_save: Optional[str] = None,
    ) -> None:
        """
        Train the model.

        Args:
            n_epochs (int): The number of epochs.
            scheduler (Optional[DictConfig], optional): The scheduler. Defaults to None.
            module_to_save (Optional[str], optional): The module to save. Defaults to None.
                If None, the whole model is saved.
        Returns:
            None
        """

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
                T_max=n_epochs * len(self.dataloaders["train"]),
                last_epoch=-1,
            )

        # Train the model
        # Set the models in training mode
        set_train_mode(model=self.model, head=self.head, module2train=self.module2train)

        # Chek if in self.prototypes_folder there are the prototypes for the used dataset
        if self.proto_train:
            proto_folder = (
                self.prototypes_folder / self.model.__class__.__name__.lower()
            )
            proto_folder.mkdir(parents=True, exist_ok=True)
            proto_path = (
                proto_folder
                / f"{self.dataloaders['train'].dataset.__class__.__name__.lower()}.pt"
            )
            if proto_path.exists():
                logger.info("Loading prototypes from %s", proto_path)
                self.prototypes = torch.load(proto_path, weights_only=True)
                logger.info("Prototypes loaded")
            else:
                logger.info("Creating prototypes...")
                with redirect_stdout(tqdml), redirect_stderr(tqdml):
                    self.prototypes = self.model.create_prototypes(
                        dataloader=self.dataloaders["train"],
                        n_classes=self.n_classes,
                        save_path=proto_path,
                    )
                logger.info("Prototypes created")

        if not self.eval:
            # Initialize the time estimator
            e_time = TimeEstimator(total=n_epochs)

            # Initialize the accuracy
            accuracy_1 = np.zeros(n_epochs)
            accuracy_5 = np.zeros(n_epochs)

            # Init last epoch
            last_epoch = 0

            # Define the name of the run with the relatives tags and params if MLFlow is used
            if list(self.data_logger.keys())[0] == "mlflow":
                if self.data_logger["mlflow"] is not None:
                    run_name = self.data_logger["mlflow"]
                else:
                    run_name = (
                        f"{self.model.__class__.__name__}"
                        + f"_{self.dataloaders['train'].dataset.__class__.__name__}"
                        + f"_pretraining_{date.now():%d-%m-%Y@%H:%M:%S}"
                    )
                tags = {
                    "dataset": self.dataloaders["train"].dataset.__class__.__name__,
                    "model": self.model.__class__.__name__,
                }
                params = {
                    "executed_command": self.data_logger["mlflow"]["executed_command"],
                    "experiment_save_folder": self.save_folder,
                    **self.data_logger["mlflow"]["config_params"],
                }
            else:
                run_name = None
                tags = None
                params = None

            # Train the model
            logger.info("Starting pre-training...")
            e_time.start()

            with (
                mlflow.start_run(
                    run_name=run_name,
                    log_system_metrics=True,
                    tags=tags,
                )
                if list(self.data_logger.keys())[0] == "mlflow"
                else nullcontext()
            ):
                # Log training parameters on MLFlow
                if (
                    self.data_logger is not None
                    and list(self.data_logger.keys())[0] == "mlflow"
                ):
                    mlflow.log_params(params)

                for epoch in range(n_epochs):
                    # Run the epoch
                    out = self._run_epoch(
                        epoch=epoch,
                        tot_epoch=n_epochs,
                        dataloader=self.dataloaders["train"],
                        log_every=self.log_every,
                        time_estimator=e_time,
                    )

                    # Save the accuracy
                    accuracy_1[epoch] = out["accuracy_1"]
                    accuracy_5[epoch] = out["accuracy_5"]

                    # Update the last epoch
                    last_epoch = epoch

                    # Save intermediate checkpoints
                    if (epoch + 1) % 10 == 0:
                        self._save_checkpoint(
                            path=self.save_folder, epoch=epoch, module=module_to_save
                        )

                    # Evaluate the model
                    if (
                        self.evaluate_every is not None
                        and (epoch + 1) % self.evaluate_every == 0
                    ):
                        self.evaluate(
                            current_epoch=epoch,
                            log_every=self.log_every,
                        )

                    # Early stopping
                    # Stop the training if the accuracy is not improving from the last 5 epochs
                    if (
                        np.all(
                            np.abs(np.diff(accuracy_1[epoch - 5 : epoch + 1])) <= 5e-4
                        )
                        and epoch > 5
                        and self.early_stopping
                    ):
                        logger.info(
                            "Early stopping at epoch %d due to lack of improvements"
                            + " in the last 5 epochs",
                            epoch,
                        )
                        break

                # Save the model
                self._save_checkpoint(
                    path=self.save_folder,
                    epoch=last_epoch,
                    last=True,
                    module=module_to_save,
                )

                # Log the final metrics
                logger.info("")
                logger.info(
                    "Pre-training completed in %s | Accuracy-1: %.2f%% | Accuracy-5: %.2f%%",
                    e_time.get_total(h_format=True),
                    accuracy_1[-1] * 100,
                    accuracy_5[-1] * 100,
                )

                # Evaluate the model if has not been evaluated during the training
                if (
                    self.evaluate_every is None
                    or (last_epoch + 1) % self.evaluate_every != 0
                ):
                    self.evaluate(current_epoch=last_epoch, log_every=self.log_every)
        else:
            self.evaluate()
