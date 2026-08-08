"""
This module contains the pretraining training loop implementation.

Classes:
    None

Functions:
    run_epoch: Run an epoch of the model.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

import os
from typing import Optional, Dict, Tuple, List, Union
from copy import deepcopy as copy

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from modules.common.auxiliary_fn import TimeEstimator
from modules.common.logger import Logger, TqdmLogger, log2logger
from modules.common.training_utils import (
    normalize_tensor,
)

# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


# Export necessary env variable
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Define wahat to export
__all__ = ["run_epoch"]


def run_epoch(
    c_model: torch.nn.Module,
    s_model: torch.nn.Module,
    heads: Dict[str, torch.nn.Module],
    epoch: int,
    dataloaders: List[torch.utils.data.DataLoader],
    time_estimator: TimeEstimator,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    kd_loss: Optional[torch.nn.Module] = None,
    loss_weight: Optional[float] = 1.0,
    kd_loss_weight: Optional[float] = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    data_logger: Optional[Union[SummaryWriter, Dict[str, str]]] = None,
    tot_epoch: Optional[int] = 0,
    log_every: Optional[int] = 5,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
    max_iters: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run an epoch of the model.

    Args:
        c_model (torch.nn.Module): The client model.
        s_model (torch.nn.Module): The server model.
        heads (Dict[str, torch.nn.Module]): The classifiers.
        epoch (int): The current epoch.
        dataloaders (List[torch.utils.data.DataLoader]): The dataloaders. First element
            is the main dataset, eventual 2nd and 3rd elements are the additional dataloaders.
        time_estimator (TimeEstimator): The time estimator.
        device (torch.device): The device to use.
        criterion (Optional[torch.nn.Module], optional): The loss function. Defaults to None.
        kd_loss (Optional[torch.nn.Module], optional): The knowledge distillation loss function.
            Defaults to None.
        loss_weight (Optional[float], optional): The weight of the loss. Defaults to 1.0.
        kd_loss_weight (Optional[float], optional): The weight of the knowledge distillation loss.
            Defaults to 1.0.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer.
            Defaults to None.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): The learning rate
            scheduler. Defaults to None.
        data_logger (Optional[Union[SummaryWriter, Dict[str, str]], optional): The data logger.
            Defaults to None.
        tot_epoch (int): The total number of epochs.
        log_every (Optional[int], optional): The number of batches to wait before logging.
        eval (bool, optional): If True, the model is in evaluation mode. Defaults to False.
    Returns:
        Dict[str, torch.Tensor]: The output metrics.
    """

    # Log the epoch
    if not eval:
        logger.info("%s Epoch %03d/%03d %s", "-" * 44, epoch + 1, tot_epoch, "-" * 44)
    else:
        logger.info("%s Val Epoch %03d %s", "-" * 44, epoch + 1, "-" * 44)

    # Use the server model as a classifier (no training just inference)
    # Initialize the predictions and labels
    s_top_k = None
    c_top_k = None
    labels = None

    s_top_k_a1 = None
    c_top_k_a1 = None
    labels_a1 = None
    stop_a1 = False

    s_top_k_a2 = None
    c_top_k_a2 = None
    labels_a2 = None
    stop_a2 = False

    # Initialize the losses
    losses = []
    kd_losses = []

    if max_iters is None:
        max_iters = len(dataloaders[0])

    # Define the batch time estiamtor
    b_time = TimeEstimator(total=max_iters)

    # instantiate additional datasets iterators
    if len(dataloaders) > 1:
        a1_iter = iter(dataloaders[1])

    if len(dataloaders) > 2:
        a2_iter = iter(dataloaders[2])

    b_time.start()
    for i, batch in enumerate(dataloaders[0]):
        # Check if we should stop training based on the maximum number of iterations
        if i >= max_iters:
            break

        b_out = _run_batch(
            c_model=c_model,
            s_model=s_model,
            heads=heads,
            batch_index=i,
            tot_batches=max_iters,
            current_epoch=epoch,
            batch=batch,
            log_every=log_every,
            time_estimator=b_time,
            dataset_name=dataloaders[0].dataset.__class__.__name__,
            device=device,
            criterion=criterion,
            kd_loss=kd_loss,
            loss_weight=loss_weight,
            kd_loss_weight=kd_loss_weight,
            optimizer=optimizer,
            scheduler=scheduler,
            data_logger=data_logger,
            eval=eval,
            is_additional=False,
        )

        s_top_k = (
            b_out["s_top_k"]
            if s_top_k is None
            else torch.cat((s_top_k, b_out["s_top_k"]))
        )
        c_top_k = (
            b_out["c_top_k"]
            if c_top_k is None
            else torch.cat((c_top_k, b_out["c_top_k"]))
        )
        labels = (
            b_out["label"] if labels is None else torch.cat((labels, b_out["label"]))
        )

        if b_out["loss"] is not None:
            losses.append(b_out["loss"].item())
        if b_out["kd_loss"] is not None:
            kd_losses.append(b_out["kd_loss"].item())

        if len(dataloaders) > 1:
            try:
                batch_a1 = next(a1_iter)
            except StopIteration:
                if not eval:
                    a1_iter = iter(dataloaders[1])
                    batch_a1 = next(a1_iter)
                else:
                    stop_a1 = True

            if not stop_a1:
                b_out_a1 = _run_batch(
                    c_model=c_model,
                    s_model=s_model,
                    heads=heads,
                    batch_index=i,
                    tot_batches=len(dataloaders[0]),
                    current_epoch=epoch,
                    batch=batch_a1,
                    log_every=log_every,
                    time_estimator=b_time,
                    dataset_name=dataloaders[1].dataset.__class__.__name__,
                    device=device,
                    criterion=criterion,
                    kd_loss=kd_loss,
                    loss_weight=loss_weight,
                    kd_loss_weight=kd_loss_weight,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data_logger=data_logger,
                    eval=eval,
                    is_additional=True,
                )

                s_top_k_a1 = (
                    b_out_a1["s_top_k"]
                    if s_top_k_a1 is None
                    else torch.concatenate((s_top_k_a1, b_out_a1["s_top_k"]))
                )
                c_top_k_a1 = (
                    b_out_a1["c_top_k"]
                    if c_top_k_a1 is None
                    else torch.concatenate((c_top_k_a1, b_out_a1["c_top_k"]))
                )
                labels_a1 = (
                    b_out_a1["label"]
                    if labels_a1 is None
                    else torch.concatenate((labels_a1, b_out_a1["label"]))
                )

        if len(dataloaders) > 2:
            try:
                batch_a2 = next(a2_iter)
            except StopIteration:
                if not eval:
                    a2_iter = iter(dataloaders[2])
                    batch_a2 = next(a2_iter)
                else:
                    stop_a2 = True

            if not stop_a2:
                b_out_a2 = _run_batch(
                    c_model=c_model,
                    s_model=s_model,
                    heads=heads,
                    batch_index=i,
                    tot_batches=len(dataloaders[0]),
                    current_epoch=epoch,
                    batch=batch_a2,
                    log_every=log_every,
                    time_estimator=b_time,
                    dataset_name=dataloaders[2].dataset.__class__.__name__,
                    device=device,
                    criterion=criterion,
                    kd_loss=kd_loss,
                    loss_weight=loss_weight,
                    kd_loss_weight=kd_loss_weight,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data_logger=data_logger,
                    eval=eval,
                    is_additional=True,
                )

                s_top_k_a2 = (
                    b_out_a2["s_top_k"]
                    if s_top_k_a2 is None
                    else torch.concatenate((s_top_k_a2, b_out_a2["s_top_k"]))
                )
                c_top_k_a2 = (
                    b_out_a2["c_top_k"]
                    if c_top_k_a2 is None
                    else torch.concatenate((c_top_k_a2, b_out_a2["c_top_k"]))
                )
                labels_a2 = (
                    b_out_a2["label"]
                    if labels_a2 is None
                    else torch.concatenate((labels_a2, b_out_a2["label"]))
                )

    # Compute the aggregate epoch accuracy
    c_accuracy_1 = torch.mean(1.0 * (c_top_k[:, 0] == labels)).item()
    c_accuracy_5 = torch.mean(
        1.0 * torch.any(c_top_k == labels.unsqueeze(1), dim=1)
    ).item()
    s_accuracy_1 = torch.mean(1.0 * (s_top_k[:, 0] == labels)).item()
    s_accuracy_5 = torch.mean(
        1.0 * torch.any(s_top_k == labels.unsqueeze(1), dim=1)
    ).item()

    if len(dataloaders) > 1:
        c_accuracy_1_a1 = torch.mean(1.0 * (c_top_k_a1[:, 0] == labels_a1)).item()
        c_accuracy_5_a1 = torch.mean(
            1.0 * torch.any(c_top_k_a1 == labels_a1.unsqueeze(1), dim=1)
        ).item()
        s_accuracy_1_a1 = torch.mean(1.0 * (s_top_k_a1[:, 0] == labels_a1)).item()
        s_accuracy_5_a1 = torch.mean(
            1.0 * torch.any(s_top_k_a1 == labels_a1.unsqueeze(1), dim=1)
        ).item()

    if len(dataloaders) > 2:
        c_accuracy_1_a2 = torch.mean(1.0 * (c_top_k_a2[:, 0] == labels_a2)).item()
        c_accuracy_5_a2 = torch.mean(
            1.0 * torch.any(c_top_k_a2 == labels_a2.unsqueeze(1), dim=1)
        ).item()
        s_accuracy_1_a2 = torch.mean(1.0 * (s_top_k_a2[:, 0] == labels_a2)).item()
        s_accuracy_5_a2 = torch.mean(
            1.0 * torch.any(s_top_k_a2 == labels_a2.unsqueeze(1), dim=1)
        ).item()

    # Update the epoch time log
    time_estimator.update()

    # Log the epoch metrics
    logger.info("")
    logger.info(
        "[EPOCH %03d SUMMARY] "
        + "[Time for epoch: %s | ETA: %s] "
        + "Accuracy: [server=%.2f%%, client=%.2f%%] | "
        + "Accuracy_a1: [server=%s, client=%s] | "
        + "Accuracy_a2: [server=%s, client=%s]",
        epoch + 1,
        time_estimator.get_last(h_format=True),
        time_estimator.estimate(h_format=True),
        s_accuracy_1 * 100,
        c_accuracy_1 * 100,
        (
            f"{(s_accuracy_1_a1 * 100):.2f}"  # pylint: disable=possibly-used-before-assignment
            if len(dataloaders) > 1
            else "N/A"
        ),
        (
            f"{(c_accuracy_1_a1 * 100):.2f}"  # pylint: disable=possibly-used-before-assignment
            if len(dataloaders) > 1
            else "N/A"
        ),
        (
            f"{(s_accuracy_1_a2 * 100):.2f}"  # pylint: disable=possibly-used-before-assignment
            if len(dataloaders) > 2
            else "N/A"
        ),
        (
            f"{(c_accuracy_1_a2 * 100):.2f}"  # pylint: disable=possibly-used-before-assignment
            if len(dataloaders) > 2
            else "N/A"
        ),
    )
    logger.info("-" * 103)

    # Log the losses and KD loss on tensorboard or MLFlow
    if data_logger is not None and not eval:
        data = {}
        data["Pretrain/Train loss"] = f"{np.mean(losses):4f}"
        data["Pretrain/Train KD_loss"] = f"{np.mean(kd_losses):4f}"

        data["Pretrain/Train Server accuracy-1"] = s_accuracy_1 * 100
        data["Pretrain/Train Server accuracy-5"] = s_accuracy_5 * 100
        data["Pretrain/Train Client accuracy-1"] = c_accuracy_1 * 100
        data["Pretrain/Train Client accuracy-5"] = c_accuracy_5 * 100

        if len(dataloaders) > 1:
            data["Pretrain/Train Server accuracy-1-a1"] = s_accuracy_1_a1 * 100
            data["Pretrain/Train Server accuracy-5-a1"] = s_accuracy_5_a1 * 100
            data["Pretrain/Train Client accuracy-1-a1"] = c_accuracy_1_a1 * 100
            data["Pretrain/Train Client accuracy-5-a1"] = c_accuracy_5_a1 * 100

        if len(dataloaders) > 2:
            data["Pretrain/Train Server accuracy-1-a2"] = s_accuracy_1_a2 * 100
            data["Pretrain/Train Server accuracy-5-a2"] = s_accuracy_5_a2 * 100
            data["Pretrain/Train Client accuracy-1-a2"] = c_accuracy_1_a2 * 100
            data["Pretrain/Train Client accuracy-5-a2"] = c_accuracy_5_a2 * 100

        log2logger(
            logger=data_logger,
            epoch=(epoch + 1),
            data=data,
        )
    elif data_logger is not None and eval:
        data = {}

        data["Pretrain/Val Server accuracy-1"] = s_accuracy_1 * 100
        data["Pretrain/Val Server accuracy-5"] = c_accuracy_5 * 100
        data["Pretrain/Val Client accuracy-1"] = c_accuracy_1 * 100
        data["Pretrain/Val Client accuracy-5"] = c_accuracy_5 * 100

        if len(dataloaders) > 1:
            data["Pretrain/Val Server accuracy-1-a1"] = s_accuracy_1_a1 * 100
            data["Pretrain/Val Server accuracy-5-a1"] = s_accuracy_5_a1 * 100
            data["Pretrain/Val Client accuracy-1-a1"] = c_accuracy_1_a1 * 100
            data["Pretrain/Val Client accuracy-5-a1"] = c_accuracy_5_a1 * 100

        if len(dataloaders) > 2:
            data["Pretrain/Val Server accuracy-1-a2"] = s_accuracy_1_a2 * 100
            data["Pretrain/Val Server accuracy-5-a2"] = s_accuracy_5_a2 * 100
            data["Pretrain/Val Client accuracy-1-a2"] = c_accuracy_1_a2 * 100
            data["Pretrain/Val Client accuracy-5-a2"] = c_accuracy_5_a2 * 100

        log2logger(
            logger=data_logger,
            epoch=(epoch + 1),
            data=data,
        )

    out = {
        "s_accuracy_1": s_accuracy_1,
        "s_accuracy_5": s_accuracy_5,
        "c_accuracy_1": c_accuracy_1,
        "c_accuracy_5": c_accuracy_5,
    }

    return out


def _run_batch(
    c_model: torch.nn.Module,
    s_model: torch.nn.Module,
    heads: Dict[str, torch.nn.Module],
    batch_index: int,
    tot_batches: int,
    current_epoch: int,
    batch: Dict[str, torch.Tensor],
    time_estimator: TimeEstimator,
    dataset_name: str,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    kd_loss: Optional[torch.nn.Module] = None,
    loss_weight: Optional[float] = 1.0,
    kd_loss_weight: Optional[float] = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    data_logger: Optional[Union[SummaryWriter, Dict[str, str]]] = None,
    log_every: Optional[int] = 5,
    eval: bool = False,  # pylint: disable=redefined-builtin
    is_additional=False,
) -> Dict[str, torch.Tensor]:
    """
    Run a batch of data through the model.

    Args:
        c_model (torch.nn.Module): The client model.
        s_model (torch.nn.Module): The server model.
        heads (Dict[str, torch.nn.Module]): The classifiers.
        batch_index (int): The index of the batch.
        tot_batches (int): The total number of batches.
        current_epoch (int): The current epoch.
        batch (Dict[str, torch.Tensor]): The batch of data.
        time_estimator (TimeEstimator): The time estimator.
        dataset_name (str): name of the dataset, needed to select the correct classifier.
        device (torch.device): The device to use.
        criterion (Optional[torch.nn.Module], optional): The loss function. Defaults to None.
        kd_loss (Optional[torch.nn.Module], optional): The knowledge distillation loss function.
            Defaults to None.
        loss_weight (Optional[float], optional): The weight of the loss. Defaults to 1.0.
        kd_loss_weight (Optional[float], optional): The weight of the knowledge distillation loss.
            Defaults to 1.0.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer.
            Defaults to None.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): The learning rate
            scheduler. Defaults to None.
        data_logger (Optional[Union[SummaryWriter, Dict[str, str]], optional): The data logger.
            Defaults to None.
        log_every (Optional[int], optional): The number of batches to wait before logging.
        eval (bool, optional): If True, the model is in evaluation mode. Defaults to False.
        is_additional (bool, optional): If True, logging is disabled. Defaults to False.
        loss_scale (float, optional): Scaling factor applied to the loss. Defaults to 1.0.
    Returns:
        Dict[str, torch.Tensor]: The output of the model.
    """

    # Move the data to device
    if "c_image" in batch:
        c_image = batch["c_image"].to(device)
        s_image = batch["s_image"].to(device)
    else:
        c_image = batch["image"].to(device)
        s_image = c_image
    label = batch["label"].to(device)
    if batch["embeddings"] is not None:
        s_feat = batch["embeddings"].to(device)
    else:
        # make sure server is always in eval mode
        s_model.eval()
        with torch.no_grad():
            # Compute the server features
            s_feat = s_model.get_class_scores(image=s_image, normalize=True)

    # set the client and classifiers
    # in eval or train depending on eval flag
    if eval:
        c_model.eval()
        clas = heads[dataset_name].eval()
    else:
        c_model.train()
        clas = heads[dataset_name].train()

    # Clone the classifier for frozen evaluation
    # we need to do this to allow the gradient
    # to flow through the classifier during training
    f_clas = _clone_frozen(clas)

    # Compute the client prediction
    with torch.set_grad_enabled(not eval):
        c_feat = c_model(c_image)

        # Normalize both client and server features
        # before computing the predictions
        s_logits = clas(normalize_tensor(s_feat))
        c_logits = f_clas(normalize_tensor(c_feat))

    s_out = {"feats": s_feat, "logits": s_logits}
    c_out = {"feats": c_feat, "logits": c_logits}

    # Get the class predictions
    s_top_k, c_top_k = _get_class_predictions(s_logits=s_logits, c_logits=c_logits)

    # Compute the loss
    if not eval:
        kd_loss, loss = _compute_loss(
            s_out=s_out,
            c_out=c_out,
            label=label,
            device=device,
            criterion=criterion,
            kd_loss=kd_loss,
            loss_weight=loss_weight,
            kd_loss_weight=kd_loss_weight,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the lr scheduler
        if scheduler is not None and not is_additional:
            scheduler.step()

    else:
        loss = None
        kd_loss = None

    # Compute the batch accuracy
    c_accuracy_1 = torch.mean(1.0 * (c_top_k[:, 0] == label)).item()
    s_accuracy_1 = torch.mean(1.0 * (s_top_k[:, 0] == label)).item()

    # Log the partial results and evaluation status
    if not is_additional:
        # Update the batch time log
        time_estimator.update()
        if (batch_index + 1) % log_every == 0:
            logger.info(
                "[Batch %03d/%03d][Time for batch: %s | ETA %s] "
                + "Batch accuracy: [server=%.2f%%, client=%.2f%%]",
                batch_index + 1,
                tot_batches,
                time_estimator.get_last(h_format=True),
                time_estimator.estimate(h_format=True),
                s_accuracy_1 * 100,
                c_accuracy_1 * 100,
            )

            # Log the batch metrics on tensorboard or MLFlow
            if data_logger is not None and not eval:
                data = {}
                if criterion is not None:
                    data["Pretrain/Train batch-loss"] = loss.item()
                if kd_loss is not None:
                    data["Pretrain/Train batch-KD_loss"] = kd_loss.item()
                if scheduler is not None:
                    data["Pretrain/Train batch-lr"] = optimizer.param_groups[0]["lr"]
                log2logger(
                    logger=data_logger,
                    epoch=(batch_index + 1) + (current_epoch * tot_batches),
                    data=data,
                )

    out = {
        "s_top_k": s_top_k,
        "c_top_k": c_top_k,
        "label": label,
        "loss": loss,
        "kd_loss": kd_loss,
    }

    return out


def _get_class_predictions(
    s_logits: torch.Tensor, c_logits: torch.Tensor
) -> Tuple[torch.Tensor]:
    """
    Get the class predictions
    it return the top-5 class prediction for the server and the client model.

    Args:
        s_logits (torch.Tensor): The output (logits) of the server classifier.
        c_logits (torch.Tensor): The output (logits) of the client classifier.

    Returns:
        Tuple[torch.Tensor]: The top-5 class predictions [s_top_k, c_top_k].
    """

    # Get the class predictions
    s_top_k = torch.topk(s_logits, k=5, dim=1)[1]
    c_top_k = torch.topk(c_logits, k=5, dim=1)[1]

    return s_top_k, c_top_k


def _clone_frozen(module: torch.nn.Module) -> torch.nn.Module:
    """
    Clone a frozen module and return it.
    Args:
        module (nn.Module): The frozen module to clone.
        Returns:
            nn.Module: The cloned module.
    """
    f_module = copy(module)
    f_module.eval()
    for param in f_module.parameters():
        param.requires_grad = False
    return f_module


def _compute_loss(
    s_out: Dict[str, torch.Tensor],
    c_out: Dict[str, torch.Tensor],
    label: torch.Tensor,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    kd_loss: Optional[torch.nn.Module] = None,
    loss_weight: Optional[float] = 1.0,
    kd_loss_weight: Optional[float] = 1.0,
) -> Tuple[torch.Tensor]:
    """
    Compute the loss of the model.
    It works both with prototypes that with standard linear classifier.
    It depends if the prototypes are provided or not.
    If prototypes are provided, the loss is computed as a combination of two losses:
    - kd_loss -> the distance between the features of the client model and the server model
    The final loss is a combination of the two losses with a weight factor.
    The weight factor is used to balance the two losses as follow:
    loss = (1 - weight) * kd_loss + weight * kd_loss_proto
    the weight parameter is self.kd_loss_weight.

    Args:
        s_out (Dict[str, torch.Tensor]): The output of the server model.
        c_out (Dict[str, torch.Tensor]): The output of the client model.
        label (torch.Tensor): The label of the batch.
        device (torch.device): The device to use.
        criterion (Optional[torch.nn.Module]): The loss function.
        kd_loss (Optional[torch.nn.Module]): The knowledge distillation loss function.
        loss_weight (Optional[float]): The weight of the loss.
        kd_loss_weight (Optional[float]): The weight of the knowledge distillation loss.
    Returns:
        Tuple[torch.Tensor]: The loss of the model [kd_loss, total_loss].
    """

    # Compute the loss
    if criterion is not None:
        lce_c = criterion(c_out["logits"], label)
        lce_s = criterion(s_out["logits"], label)
        loss = (
            0.5 * lce_c + 0.5 * lce_s
        )  # cross entropy balanced between client and server models
    else:
        loss = torch.zeros(1).to(device)

    if kd_loss is not None:
        kd_loss = kd_loss(c_out["feats"], s_out["feats"])
    else:
        kd_loss = torch.zeros(1).to(device)

    # Combine the two losses
    loss = loss_weight * loss + kd_loss_weight * kd_loss

    return kd_loss, loss
