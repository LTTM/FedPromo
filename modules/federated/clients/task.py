"""
This module contains the function of the training and evaluating loop
of the federated learning process.

Classes:
    None

Functions:
    train: Train the network on the training set.
    test: Evaluate the network on the entire test set.

Constants:
    None

Exceptions:
    None


Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union, Dict, Any, Optional, Literal
from copy import deepcopy as copy

import torch

from modules.common.training_utils import (
    set_train_mode,
    set_eval_mode,
    normalize_tensor,
)


_all_ = ["train", "test"]


def train(
    models: Dict[str, torch.nn.Module],
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    epochs: int,
    train_config: Dict[Literal["base", "moon"], Any],
    module2train: str = "all",
) -> Dict[str, Any]:
    """
    Train the network on the training set.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        head (torch.nn.Module): The head of the model.
        device (Union[torch.device, str]): The device on which the training will be performed.
        loss (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        epochs (int): The number of epochs.
        train_config (Dict[Literal["base", "moon"], Any]): The training configuration.
            Contains the type of training and any additional arguments needed for the
            specific type of training
        module2train (str): The module to train.

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Cast the device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Extract the basic models
    model = models["model"]
    head = models["head"]

    # Set the modules in training or eval mode according to the module2train
    set_train_mode(model=model, head=head, module2train=module2train)

    # Define the losses and accuracies accumulators
    tot_loss, tot_acc = torch.zeros(epochs), torch.zeros(epochs)

    # Train the model
    for epoch in range(epochs):
        # Run an epoch
        metrics = _run_epoch(
            models=models,
            device=device,
            dataloader=dataloader,
            epoch=epoch,
            criterion=criterion,
            optimizer=optimizer,
            train_config=train_config,
        )

        if train_config["class_debias"]:
            # Center the distribution of the classifier parameters
            with torch.no_grad():
                models["head"].fcn[-1].weight -= train_config["class_debias_weight"]*(
                    models["head"].fcn[-1].weight.mean(dim=0, keepdim=True)
                )

        # Update the metrics
        tot_loss[epoch] = metrics["loss"]
        tot_acc[epoch] = metrics["accuracy"]

    # Compute the total loss and accuracy
    tot_loss = tot_loss.mean().item()
    tot_acc = tot_acc.mean().item()

    # Print the total metrics
    print(f"[Train results] - train {tot_loss:1.4f} | accuracy {(tot_acc * 100):6.2f}%")

    # Define the output metrics
    metrics = {"loss": tot_loss, "accuracy": tot_acc}

    return metrics


def test(
    models: Dict[str, torch.nn.Module],
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> Dict[str, Any]:
    """
    Evaluate the network on the entire test set.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (Union[torch.device, str]): The device on which the training will be performed.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The test data loader.

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Cast the device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Extartc the basic models
    model = models["model"]
    head = models["head"]

    # Set the model in evaluation mode
    set_eval_mode(model=model, head=head)

    # Evaluate the model
    metrics = _run_epoch(
        models=models,
        device=device,
        dataloader=dataloader,
        epoch=0,
        criterion=criterion,
        eval=True,
    )

    # Extract the total loss and accuracy
    loss = metrics["loss"]
    acc = metrics["accuracy"]

    # Print the total metrics
    print(f"[Test results] - test loss {loss:1.4f} | accuracy {(acc * 100):6.2f}%")

    # Define the output metrics
    metrics = {"loss": loss, "accuracy": acc}

    return metrics


def _run_epoch(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    criterion: torch.nn.Module,
    train_config: Optional[Dict[Literal["base", "moon", "ir"], Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
) -> Dict[str, Any]:
    """
    Run an epoch of training on the client model.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (str): The device on which the training will be performed.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        epoch (int): The epoch number.
        criterion (torch.nn.Module): The loss function.
        train_config (Optional[Dict[Literal["base", "moon"], Any]]):
            The training configuration.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Define the losses and accuracies accumulators
    d_len = max(1, len(dataloader))
    epoch_loss, epoch_acc = torch.zeros(d_len), torch.zeros(d_len)

    # Define the batch function
    if train_config is not None:
        if train_config["type"] == "moon":
            batch_fn = _run_moon_batch
        elif train_config["type"] == "icp":
            batch_fn = _run_icp_batch
        elif train_config["type"] == "fedprox":
            batch_fn = _run_fedprox_batch
        else:
            batch_fn = _run_batch
    else:
        batch_fn = _run_batch

    # Initialize the label counts metric that will be used for EMA
    label_counts = torch.zeros(models["head"].fcn[-1].weight.size(0))

    # Train the model
    for i, batch in enumerate(dataloader):
        # Update the label counts
        label_counts[batch["label"]] += 1

        # Run the batch
        metrics = batch_fn(
            models=models,
            device=device,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            eval=eval,
            batch_idx=i + 1,
            tot_batches=len(dataloader),
            label_counts=label_counts,
            **train_config,
        )

        # Update the metrics
        epoch_loss[i] = metrics["loss"]
        epoch_acc[i] = metrics["accuracy"]

    # Compute the epoch loss and accuracy
    if len(epoch_loss) > 1:
        epoch_loss = epoch_loss[-1].item()
        epoch_acc = epoch_acc[-1].item()
    else:
        epoch_loss = epoch_loss.item()
        epoch_acc = epoch_acc.item()

    # Print the epoch metrics
    if not eval:
        print(
            f"Epoch {epoch+1}: loss {epoch_loss:1.4f} | accuracy {(epoch_acc * 100):6.2f}%"
        )

    # Define the output metrics
    metrics = {"loss": epoch_loss, "accuracy": epoch_acc}

    return metrics


def _run_batch(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
    **kwargs,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    Run a batch of training on the client model.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (str): The device on which the training will be performed.
        batch (Dict[str, torch.Tensor]): The batch of data.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Move the data to the device
    images, labels = batch["image"], batch["label"]
    images, labels = images.to(device), labels.to(device)

    # Extract the basic models
    model = models["model"]
    head = models["head"]

    # Compute the network prediction
    if not eval:
        optimizer.zero_grad()
    with torch.set_grad_enabled(not eval):
        with torch.set_grad_enabled(model.training):
            feats = model(images)
        with torch.set_grad_enabled(head.training):
            outputs = head(normalize_tensor(feats))

    # Compute the loss
    loss = criterion(outputs, labels)

    if not eval:
        # Perform the backward pass
        loss.backward()
        optimizer.step()

    # Get the class predictions
    pred = torch.argmax(outputs, dim=1)

    # Compute the batch accuracy
    accuracy = torch.mean(1.0 * (pred == labels)).item()

    # Define the output metrics
    metrics = {"loss": loss.item(), "accuracy": accuracy}

    return metrics


def _run_icp_batch(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    batch_idx: int,
    tot_batches: int,
    label_counts: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
    ema_decay: Optional[float] = 0.5,
    **kwargs,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    Run a batch of training on the client model.
    Perform the training using the selective exponential moving average.
    The avareging is performed on the model weights only at the last batch
    so once per epoch.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (str): The device on which the training will be performed.
        batch (Dict[str, torch.Tensor]): The batch of data.
        batch_idx (int): The batch index.
        tot_batches (int): The total number of batches.
        label_counts (torch.Tensor): The label counts tensor.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False
        ema_decay (float): The exponential moving average decay factor.
            Default: 0.5

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Move the data to the device
    images, labels = batch["image"], batch["label"]
    images, labels = images.to(device), labels.to(device)

    # Compute the network prediction
    if not eval:
        optimizer.zero_grad()
    with torch.set_grad_enabled(not eval):
        with torch.set_grad_enabled(models["model"].training):
            feats = models["model"](images)
        with torch.set_grad_enabled(models["head"].training):
            outputs = models["head"](normalize_tensor(feats))

    # Compute the loss
    loss = criterion(outputs, labels)

    if not eval:
        # Perform the backward pass
        loss.backward()
        optimizer.step()

    # Get the class predictions
    pred = torch.argmax(outputs, dim=1)

    # Compute the batch accuracy
    accuracy = torch.mean(1.0 * (pred == labels)).item()

    # Define the output metrics
    metrics = {"loss": loss.item(), "accuracy": accuracy}

    # Update the EMA model
    # if batch_idx == tot_batches:
    if batch_idx == tot_batches:
        # Extract inactive classes
        inactive_classes = label_counts == 0

        # Apply ICP: EMA on all layers, except the last, which applies parameter isolation
        with torch.no_grad():
            # Extract target device
            device = models["head"].fcn[-1].weight.device

            # Last layer of classifier has EMA + Parameter Isolation
            models["head"].fcn[-1].weight[inactive_classes] = (
                ema_decay * models["head"].fcn[-1].weight[inactive_classes]
                + (1 - ema_decay) * models["ema_head"].fcn[-1].weight[inactive_classes].to(device=device)
            )

            # Rest of classifier has normal EMA
            for ih in range(len(models["head"].fcn[:-1])):
                cur_state_dict = models["head"].fcn[ih].state_dict()
                ema_state_dict = models["ema_head"].fcn[ih].state_dict()
                for key in cur_state_dict:
                    cur_state_dict[key] = ema_decay * cur_state_dict[key] + (1 - ema_decay) * ema_state_dict[key].to(device=device)
                models["head"].fcn[ih].load_state_dict(cur_state_dict)

            # Encoder has normal EMA
            cur_state_dict = models["model"].state_dict()
            ema_state_dict = models["ema_model"].state_dict()
            for key in cur_state_dict:
                cur_state_dict[key] = ema_decay * cur_state_dict[key] + (1 - ema_decay) * ema_state_dict[key].to(device=device)
            models["model"].load_state_dict(cur_state_dict)

        # Update the EMA model
        models["ema_head"] = copy(models["head"]).cpu()
        models["ema_model"] = copy(models["model"]).cpu()

        # Empty CUDA cache at the end
        torch.cuda.empty_cache()

    return metrics


def _run_moon_batch(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    mu: float = 0.9,
    temperature: float = 1.0,
    **kwargs,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    Run a batch of training on the client model.
    Perform the training using the moon training method.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (str): The device on which the training will be performed.
        batch (Dict[str, torch.Tensor]): The batch of data.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        mu (float): The momentum factor.
            Default: 0.9
        temperature (float): The temperature factor.
            Default: 1.0

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Move the data to the device
    images, labels = batch["image"], batch["label"]
    images, labels = images.to(device), labels.to(device)

    # Extract the models
    model = models["model"]
    prev_model = models["prev_model"]
    head = models["head"]
    prev_head = models["prev_head"]
    global_model = models["global_model"]
    global_head = models["global_head"]

    # Set all the models in eval mode
    set_eval_mode(model=prev_model, head=prev_head)
    set_eval_mode(model=global_model, head=global_head)

    # Compute the network prediction
    optimizer.zero_grad()

    with torch.set_grad_enabled(model.training):
        feats = model(images)
    with torch.set_grad_enabled(head.training):
        outputs = head(normalize_tensor(feats))
    with torch.no_grad():
        prev_feats = prev_model(images)
        prev_outputs = prev_head(normalize_tensor(prev_feats))
        global_feats = global_model(images)
        global_outputs = global_head(normalize_tensor(global_feats))

    pos = torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
        outputs, global_outputs.detach(), dim=-1
    )
    neg = torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
        outputs, prev_outputs.detach(), dim=-1
    )

    moon_x = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
    moon_x /= temperature

    moon_y = torch.zeros(images.size(0)).long().to(device)

    # Compute the loss
    out_loss = criterion(outputs, labels)
    moon_loss = mu * criterion(moon_x, moon_y)

    loss = out_loss + moon_loss

    # Perform the backward pass
    loss.backward()
    optimizer.step()

    # Get the class predictions
    pred = torch.argmax(outputs, dim=1)

    # Compute the batch accuracy
    accuracy = torch.mean(1.0 * (pred == labels)).item()

    # Define the output metrics
    metrics = {"loss": loss.item(), "accuracy": accuracy}

    return metrics


def _in_checker(
    optimizer: torch.optim.Optimizer,
    eval: bool = False,  # pylint: disable=redefined-builtin
) -> None:
    """
    Check if the input data are correct.
    if not in eval mode the loss function and the optimizer are required.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        None
    """

    if not eval and optimizer is None:
        raise ValueError(
            "Since the network is in training mode, the optimizer is required."
        )

    return None


def _run_fedprox_batch(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    batch_idx: int,
    tot_batches: int,
    label_counts: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
    proximal_mu: Optional[float] = 1.0,
    **kwargs,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    Run a batch of training on the client model.
    Perform the training using the fedprox algorithm.

    Args:
        models (Dict[str, torch.nn.Module]): The client model.
        device (str): The device on which the training will be performed.
        batch (Dict[str, torch.Tensor]): The batch of data.
        batch_idx (int): The batch index.
        tot_batches (int): The total number of batches.
        label_counts (torch.Tensor): The label counts tensor.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False
        proximal_mu (float): The proximal term factor.
            Default: 0.5

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Move the data to the device
    images, labels = batch["image"], batch["label"]
    images, labels = images.to(device), labels.to(device)

    # Compute the network prediction
    if not eval:
        optimizer.zero_grad()
    with torch.set_grad_enabled(not eval):
        with torch.set_grad_enabled(models["model"].training):
            feats = models["model"](images)
        with torch.set_grad_enabled(models["head"].training):
            outputs = models["head"](normalize_tensor(feats))

    # Compute the proximal term
    if not eval:
        model_proximal_term = 0.0
        head_proximal_term = 0.0
        if models["model"].training:
            global_model_param = [
                val.detach().clone() for val in models["last_round_model"].parameters()
            ]
            for local_weights, global_weights in zip(
                models["model"].parameters(), global_model_param
            ):
                model_proximal_term += torch.square(
                    (local_weights - global_weights).norm(2)
                )
        if models["head"].training:
            global_head_param = [
                val.detach().clone() for val in models["last_round_head"].parameters()
            ]
            for local_weights, global_weights in zip(
                models["head"].parameters(), global_head_param
            ):
                head_proximal_term += torch.square(
                    (local_weights - global_weights).norm(2)
                )

    # Compute the loss
    loss = criterion(outputs, labels) + (proximal_mu / 2) * (
        model_proximal_term + head_proximal_term
    )

    if not eval:
        # Perform the backward pass
        loss.backward()
        optimizer.step()

    # Get the class predictions
    pred = torch.argmax(outputs, dim=1)

    # Compute the batch accuracy
    accuracy = torch.mean(1.0 * (pred == labels)).item()

    # Define the output metrics
    metrics = {"loss": loss.item(), "accuracy": accuracy}

    return metrics
